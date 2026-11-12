"""R-LWE parameter benchmarks: speed vs. estimated security vs. key-mismatch rate.

A developer tool, not part of the installed package. It measures candidate (n, q, k) sets,
estimates their security and failure rate, and selects the set written to
lwe_sftp/rlwe_params.json, which both ends of the protocol then load.

    python tools/bench.py sweep                  # sweep (n, q, k), print a markdown table + the selected set
    python tools/bench.py sweep --write          # ...and write it to lwe_sftp/rlwe_params.json
    python tools/bench.py sweep --csv out.csv    # ...and dump every measured row
    python tools/bench.py validate               # check the estimators (NewHope numbers, measured failures)
    python tools/bench.py handshake [n,q,k ...]  # time the complete handshake and measure its size

Security: "core-SVP" cost of the primal uSVP attack (the 2016 estimate used by NewHope/Kyber):
smallest BKZ block size b with  sigma*sqrt(b) <= delta_b^(2b-d-1) * q^(m/d),  d = n+m+1,
minimised over m <= n samples; cost 2^(0.292 b) classical, 2^(0.265 b) quantum. The '2e'
errors don't help: multiplying a public key by 2^-1 mod q gives a standard instance with
error sigma = sqrt(k/2). This is a well-known rough estimate (it ignores the dual attack and
hybrid/algebraic attacks), good for ranking parameter sets, not for a formal claim.

Failure: v_i - v_r = 2(e_r s_i - e_i s_r + e'_i - e'_r), variance 4(2n sigma^4 + 2 sigma^2).
A coefficient disagrees only if that difference wraps past +-q/2 from a point roughly uniform
in [-q/4, q/4]; summed over the n coefficients (Gaussian approximation).
"""

import argparse
import csv
import json
import math
import os
import socket
import sys
import threading
import time

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lwe_sftp import rlwe, transport  # noqa: E402
from lwe_sftp.rlwe import Params  # noqa: E402

TCP_INITIAL_WINDOW = 10 * 1448  # Linux initcwnd 10 segments: bytes a peer can send in its first round trip


# --- security estimate ------------------------------------------------------

def _delta(b):
    """Root-Hermite factor delta_b that BKZ with block size b achieves (standard asymptotic formula)."""
    return ((math.pi * b) ** (1 / b) * b / (2 * math.pi * math.e)) ** (1 / (2 * (b - 1)))


def primal_blocksize(n, q, sigma):
    """Smallest BKZ block size b for which the primal uSVP attack succeeds, or None.

    For each number of samples m from n/8 to n (in steps of n/64), scans b upward and takes the
    first that satisfies the inequality in the module docstring; returns the smallest over all m.
    """
    best = None
    for m in range(max(n // 8, 16), n + 1, max(n // 64, 1)):
        d = n + m + 1
        for b in range(50, d):
            if sigma * math.sqrt(b) <= _delta(b) ** (2 * b - d - 1) * q ** (m / d):
                if best is None or b < best:
                    best = b
                break
    return best


def security(P):
    """Core-SVP estimate for P. Returns (b, classical bits, quantum bits).

    Feeds the error width sigma = sqrt(k/2) to primal_blocksize; the costs are 0.292 b and 0.265 b
    bits. Returns (None, 0, 0) if no block size works.
    """
    b = primal_blocksize(P.n, P.q, math.sqrt(P.k / 2))
    if b is None:
        return None, 0.0, 0.0
    return b, 0.292 * b, 0.265 * b


# --- failure estimate -------------------------------------------------------

def log2_failure(P):
    """Analytic log2 probability that an exchange disagrees on at least one key bit.

    Models each coefficient's difference v_i - v_r as Gaussian with variance
    4(2n sigma^4 + 2 sigma^2). Takes the expected overshoot past q/4, divided by q/2, as the
    per-coefficient failure probability, and multiplies by n (union bound).
    """
    s2 = P.k / 2
    var = 4 * (2 * P.n * s2 * s2 + 2 * s2)
    sd = math.sqrt(var)
    a = P.q / 4
    # E[max(0, |d| - q/4)] for d ~ N(0, sd^2), divided by (q/2)
    z = a / sd
    tail_mean = 2 * (sd * math.exp(-z * z / 2) / math.sqrt(2 * math.pi) - a * 0.5 * math.erfc(z / math.sqrt(2)))
    p_coef = min(1.0, max(tail_mean, 0.0) / (P.q / 2))
    p = min(1.0, P.n * p_coef)
    return math.log2(p) if p > 0 else -math.inf


def empirical_failure(P, trials):
    """Run trials full exchanges with P and return the fraction whose key bits disagree."""
    bad = 0
    for _ in range(trials):
        m, s, pub = rlwe.initiate(P)
        pr, w, kr = rlwe.respond(m, pub, P)
        bad += not np.array_equal(rlwe.finish(s, pr, w, P), kr)
    return bad / trials


def _fmt_fail(lf):
    """Format a log2 failure rate, collapsing astronomically small values to '< -1000'."""
    return "< -1000" if lf < -1000 else f"{lf:.0f}"


# --- timing and size --------------------------------------------------------

def time_exchange(P, budget=0.4):
    """Median wall time in ms of one full exchange (initiate, respond, finish) with P.

    Warms up once, then repeats for budget seconds and at least 20 times.
    """
    def once():
        """One complete R-LWE exchange, both roles in this process."""
        m, s, pub = rlwe.initiate(P)
        pr, w, _ = rlwe.respond(m, pub, P)
        rlwe.finish(s, pr, w, P)
    once()
    runs = []
    deadline = time.perf_counter() + budget
    while time.perf_counter() < deadline or len(runs) < 20:
        t0 = time.perf_counter()
        once()
        runs.append(time.perf_counter() - t0)
    return float(np.median(runs)) * 1e3


def hello_bytes(P):
    """Size of the client hello on the wire (JSON + base64), matching lwe_sftp/transport.py.

    Frame length prefix plus fixed JSON overhead, plus the base64 sizes of the X25519 key, the
    ML-KEM-768 encapsulation key and two R-LWE polynomials.
    """
    b64 = lambda n: 4 * ((n + 2) // 3)  # noqa: E731
    return 4 + 120 + len(P.name) + b64(32) + b64(1184) + 2 * b64(P.poly_bytes)


class _CountingSocket:
    """Socket wrapper that counts the bytes sent through it, to measure handshake traffic."""
    def __init__(self, sock):
        """Wrap sock with a zeroed byte counter."""
        self.sock, self.sent = sock, 0

    def sendall(self, data):
        """Count the bytes, then send them."""
        self.sent += len(data)
        self.sock.sendall(data)

    def recv(self, n):
        """Pass through to the wrapped socket."""
        return self.sock.recv(n)


def _one_handshake(P, host_key):
    """Run one real client/server handshake over a socketpair, the server in a thread.

    Returns (seconds, client-to-server bytes, server-to-client bytes) and re-raises any server error.
    """
    a, b = socket.socketpair()
    ca, cb = _CountingSocket(a), _CountingSocket(b)
    err = []

    def server():
        """Server half, run in the thread; stores any exception for the caller."""
        try:
            transport.server_handshake(cb, host_key, P)
        except Exception as e:  # surfaced below
            err.append(e)

    th = threading.Thread(target=server)
    t0 = time.perf_counter()
    th.start()
    transport.client_handshake(ca, P)
    th.join()
    dt = time.perf_counter() - t0
    a.close()
    b.close()
    if err:
        raise err[0]
    return dt, ca.sent, cb.sent


def bench_handshake(P, runs=300):
    """Time runs handshakes with one host key after a warm-up.

    Returns (median ms, 95th-percentile ms, client-to-server bytes, server-to-client bytes).
    """
    key = Ed25519PrivateKey.generate()
    _one_handshake(P, key)
    times = []
    for _ in range(runs):
        dt, c2s, s2c = _one_handshake(P, key)
        times.append(dt)
    return float(np.median(times)) * 1e3, float(np.percentile(times, 95)) * 1e3, c2s, s2c


# --- sweep and selection ----------------------------------------------------

QS = [3329, 7681, 12289, 40961, 65537, 786433, 8380417]
NS = [256, 512, 1024, 2048]
KS = [1, 2, 4, 8, 16]


def sweep():
    """Measure every (n, q, k) in NS x QS x KS that Params accepts. Returns a list of row dicts.

    Sets that fail more often than 2^-20 are skipped before timing. Each remaining set gets its
    exchange time, failure estimate, security estimate and hello size; progress goes to stderr.
    """
    rows = []
    for n in NS:
        for q in QS:
            for k in KS:
                try:
                    P = Params(n, q, k)
                except ValueError:
                    continue
                lf = log2_failure(P)
                if lf > -20:  # fails more than ~1 in a million handshakes: not usable, skip timing
                    continue
                b, cls, qnt = security(P)
                rows.append(dict(n=n, q=q, k=k, ms=time_exchange(P), log2_fail=lf, bkz=b,
                                 classical=cls, quantum=qnt, hello_bytes=hello_bytes(P)))
                print(f"  {P.name:<26} {rows[-1]['ms']:.3f} ms  core-SVP {cls:5.0f}  fail 2^{lf:.0f}",
                      file=sys.stderr)
    return rows


def select(rows, args):
    """Fastest-ish set that meets every constraint; among near-ties, the most secure one.

    Keeps rows meeting the four thresholds (security, failure, k, hello size), sorts them by speed,
    treats every row within tolerance of the fastest as a tie, and picks the tie with the highest
    classical security, then the lowest failure. Returns (chosen row or None, qualifying rows).
    """
    ok = [r for r in rows
          if r["classical"] >= args.min_classical and r["log2_fail"] <= args.max_log2_fail
          and r["k"] >= args.min_k and r["hello_bytes"] <= args.max_hello_bytes]
    if not ok:
        return None, ok
    ok.sort(key=lambda r: r["ms"])
    near = [r for r in ok if r["ms"] <= ok[0]["ms"] * (1 + args.tolerance)]
    return max(near, key=lambda r: (r["classical"], -r["log2_fail"])), ok


# --- commands ---------------------------------------------------------------

def cmd_validate(args):
    """'validate': print the estimators next to reference values.

    The security estimator is compared with NewHope's published figure, and the failure model with
    measured failure rates for sets chosen to fail often enough to measure.
    """
    print("Estimator vs. NewHope paper (primal, classical core-SVP):")
    for n, q, k, paper in [(1024, 12289, 16, 282), (512, 12289, 8, None)]:
        b, cls, qnt = security(Params(n, q, k))
        print(f"  n={n} q={q} k={k}: b={b}, classical {cls:.0f}, quantum {qnt:.0f}"
              + (f"   (paper: b=967, classical {paper})" if paper else ""))
    print("\nFailure model vs. measurement (parameter sets chosen to fail often):")
    for P, trials in [(Params(512, 3329, 8), 3000), (Params(2048, 3329, 4), 3000),
                      (Params(2048, 7681, 8), 3000), (Params(2048, 12289, 16), 3000)]:
        print(f"  {P.name:<24} predicted {2 ** log2_failure(P):.4f}   measured "
              f"{empirical_failure(P, trials):.4f} ({trials} trials)")


def cmd_sweep(args):
    """'sweep': run the sweep, report it, and optionally save the selection.

    Optionally dumps every row to CSV. Prints a markdown table of qualifying sets (speed relative
    to the current default, security, failure, hello size) marking the current default and the
    selection. With --write, saves the selected set with its constraints and estimates as the
    parameters file. Exits 1 if no set qualifies.
    """
    rows = sweep()
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    best, ok = select(rows, args)
    cur = next((r for r in rows if Params(r["n"], r["q"], r["k"]) == rlwe.DEFAULT), None)
    ref = cur or (ok[0] if ok else rows[0])
    print(f"\nSets meeting: core-SVP >= {args.min_classical:.0f}, failure <= 2^{args.max_log2_fail:.0f}, "
          f"k >= {args.min_k}, hello <= {args.max_hello_bytes} B. Fastest first; "
          "'*' = current default, '>' = selected.\n")
    print("| n | q | k | exchange ms | vs current | core-SVP classical / quantum | log2 failure | hello bytes |")
    print("|---|---|---|---|---|---|---|---|")
    for r in ok:
        mark = (" *" if r is cur else "") + (" >" if r is best else "")
        print(f"| {r['n']} | {r['q']} | {r['k']}{mark} | {r['ms']:.3f} | {ref['ms'] / r['ms']:.2f}x | "
              f"{r['classical']:.0f} / {r['quantum']:.0f} | {_fmt_fail(r['log2_fail'])} | {r['hello_bytes']} |")
    if best is None:
        print("\nnothing meets the constraints", file=sys.stderr)
        sys.exit(1)
    print(f"\nselected: n={best['n']}, q={best['q']}, k={best['k']}")

    if args.write:
        record = {
            "n": best["n"], "q": best["q"], "k": best["k"],
            "selected_by": "tools/bench.py sweep --write",
            "date": time.strftime("%Y-%m-%d"),
            "constraints": {"min_classical_core_svp": args.min_classical, "max_log2_failure": args.max_log2_fail,
                            "min_k": args.min_k, "max_hello_bytes": args.max_hello_bytes,
                            "tie_tolerance": args.tolerance},
            "estimates": {"core_svp_classical": round(best["classical"]), "core_svp_quantum": round(best["quantum"]),
                          "bkz_blocksize": best["bkz"], "log2_failure": _fmt_fail(best["log2_fail"]),
                          "exchange_ms": round(best["ms"], 3), "hello_bytes": best["hello_bytes"]},
        }
        with open(args.write, "w") as f:
            json.dump(record, f, indent=2)
            f.write("\n")
        print(f"wrote {args.write}")


def cmd_handshake(args):
    """'handshake': benchmark full handshakes for the given sets, or a default list.

    Prints a markdown table of median and p95 times, bytes each way, and whether the client hello
    fits in the TCP initial window.
    """
    sets = [Params.parse(a) for a in args.sets] or [
        rlwe.DEFAULT, Params(2048, 8380417, 2), Params(2048, 12289, 2), Params(1024, 12289, 4), Params(1024, 12289, 2),
    ]
    print("| R-LWE params | handshake median ms | p95 ms | client->server bytes | server->client bytes | "
          "hello fits TCP initial window |")
    print("|---|---|---|---|---|---|")
    for P in sets:
        med, p95, c2s, s2c = bench_handshake(P)
        print(f"| {P.name}{' (default)' if P == rlwe.DEFAULT else ''} | {med:.2f} | {p95:.2f} | {c2s} | {s2c} | "
              f"{'yes' if c2s <= TCP_INITIAL_WINDOW else 'no, +1 RTT'} |")


def main():
    """Parse the subcommand and its options, then dispatch to the matching cmd_* function."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sw = sub.add_parser("sweep", help="sweep parameter sets and select one")
    sw.add_argument("--csv", help="also dump every measured row to this CSV")
    sw.add_argument("--write", nargs="?", const=rlwe.PACKAGED_PARAMS, metavar="PATH",
                    help="write the selected set as the parameters file (default: lwe_sftp/rlwe_params.json)")
    sw.add_argument("--min-classical", type=float, default=128, help="core-SVP floor, classical bits (default 128)")
    sw.add_argument("--max-log2-fail", type=float, default=-64, help="key-mismatch ceiling, log2 (default -64)")
    sw.add_argument("--min-k", type=int, default=2,
                    help="smallest CBD k (default 2; k=1 secrets invite attacks the primal estimate ignores)")
    sw.add_argument("--max-hello-bytes", type=int, default=TCP_INITIAL_WINDOW,
                    help=f"client hello must fit this many bytes (default {TCP_INITIAL_WINDOW}: TCP's initial "
                         "window, so the handshake needs no extra round trip)")
    sw.add_argument("--tolerance", type=float, default=0.15,
                    help="sets within this fraction of the fastest count as ties; ties go to the most secure "
                         "(default 0.15)")

    sub.add_parser("validate", help="check the security and failure estimators")

    hs = sub.add_parser("handshake", help="time the complete handshake for parameter sets")
    hs.add_argument("sets", nargs="*", metavar="N,Q,K")

    args = ap.parse_args()
    {"sweep": cmd_sweep, "validate": cmd_validate, "handshake": cmd_handshake}[args.cmd](args)


if __name__ == "__main__":
    main()
