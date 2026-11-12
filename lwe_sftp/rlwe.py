"""Ring-LWE key exchange after Ding, Xie & Lin (IACR ePrint 2012/688).

One of the three key exchanges transport.py combines into the session key. This module provides:
  * Params, a validated (n, q, k) parameter set, and resolve_params(), which picks the set in
    use from a flag, the environment, a config file or the packaged rlwe_params.json;
  * arithmetic in R_q = Z_q[x] / (x^n + 1), with an exact FFT-based multiplication;
  * samplers for uniform and small (centered binomial) polynomials, all fed by os.urandom;
  * Ding's reconciliation (Sig / Mod2) and the three protocol roles initiate / respond / finish;
  * the wire encoding of polynomials and hint bits.

Flow (the initiator picks the public polynomial m):
    initiator:  m <- uniform, s_i, P_i = m*s_i + 2e_i              -> (m, P_i)
    responder:  s_r, P_r = m*s_r + 2e_r
                v_r = P_i*s_r + 2e'_r,  w = Sig(v_r),  k = Mod2(v_r, w)  -> (P_r, w)
    initiator:  v_i = P_r*s_i + 2e'_i,  k = Mod2(v_i, w)

v_i and v_r are both close to m*s_i*s_r, differing only by small error terms; the hint w tells
the initiator which coefficients lie near the wrap-around so both round them the same way.

Parameters are a Params value so they can be benchmarked and swept (see tools/bench.py). DEFAULT
is the set the protocol uses unless overridden. Both peers must agree on it: its name is sent in
the client hello and the server refuses a mismatch.
"""

import json
import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Params:
    """An R-LWE parameter set (n, q, k), validated on construction.

    Frozen, so it can be compared and used as a dictionary key. The wire sizes are derived
    properties, so the encoding follows the parameters automatically.
    """
    n: int   # ring degree, power of two
    q: int   # odd modulus (FFT multiplication means it need not be NTT-friendly)
    k: int   # centered-binomial parameter for secrets/errors: B_k, variance k/2

    def __post_init__(self):
        """Reject sets the implementation can't handle, raising ValueError.

        n must be a power of two (so the zero-padded FFT length 2n is one too), q must be odd, and
        k >= 1. The exact product of a centered polynomial and a small one is bounded by n*(q/2)*k;
        that has to stay well under float64's 53-bit mantissa (here: below 2^45) for poly_mul's
        rounding to recover the integers exactly.
        """
        if self.n < 2 or self.n & (self.n - 1):
            raise ValueError("n must be a power of two")
        if self.q < 5 or self.q % 2 == 0:
            raise ValueError("q must be odd")
        if self.k < 1:
            raise ValueError("k must be >= 1")
        # FFT products are exact while |coefficients| stay far below 2^53
        if self.n * (self.q // 2) * self.k >= 2 ** 45:
            raise ValueError("parameters too large for exact float64 FFT multiplication")

    @property
    def name(self):
        """Canonical text form 'n=..,q=..,k=..'.

        Sent in the client hello and compared verbatim by the server; parse() accepts it back.
        """
        return f"n={self.n},q={self.q},k={self.k}"

    @property
    def q_half(self):
        """(q-1)/2: the edge of the centered range [-(q-1)/2, (q-1)/2] and the shift Mod2 adds for hint bit 1."""
        return (self.q - 1) // 2

    @property
    def coef_dtype(self):
        """NumPy wire dtype for one coefficient: little-endian uint16 when q fits in 16 bits, else uint32.

        Halving the width is what keeps the client hello inside TCP's initial congestion window.
        """
        return "<u2" if self.q <= 1 << 16 else "<u4"

    @property
    def poly_bytes(self):
        """Encoded size of one polynomial: n coefficients times the coefficient width."""
        return self.n * np.dtype(self.coef_dtype).itemsize

    @property
    def hint_bytes(self):
        """Encoded size of the reconciliation hint: one bit per coefficient, rounded up to whole bytes."""
        return (self.n + 7) // 8

    @classmethod
    def parse(cls, text):
        """Parse '1024,12289,4' or 'n=1024,q=12289,k=4' (the form Params.name produces).

        Splits on commas. If every part is key=value, reads n, q and k by name; otherwise expects
        exactly three bare integers. The constructor re-validates the result. Any failure is re-raised
        as one ValueError that names the offending text.
        """
        parts = [x.strip() for x in str(text).split(",")]
        try:
            if all("=" in x for x in parts):
                kv = dict(x.split("=", 1) for x in parts)
                return cls(int(kv["n"]), int(kv["q"]), int(kv["k"]))
            n, q, k = map(int, parts)
            return cls(n, q, k)
        except (ValueError, KeyError) as e:
            raise ValueError(f"bad R-LWE parameters {text!r} (expected n,q,k): {e}") from None

    @classmethod
    def load(cls, path):
        """Read a parameters file as written by tools/bench.py sweep --write.

        The file is {"n": .., "q": .., "k": .., ...metadata}. Only n, q and k are used; the rest
        (constraints, estimates, date) is a record for humans. Missing or malformed values raise
        ValueError naming the file.
        """
        with open(path) as f:
            data = json.load(f)
        try:
            return cls(int(data["n"]), int(data["q"]), int(data["k"]))
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"{path}: invalid R-LWE parameters: {e}") from None


# Parameters come from data, not code: tools/bench.py sweep --write selects them and writes
# PACKAGED_PARAMS, which ships with the package. resolve_params() layers overrides on top.
PACKAGED_PARAMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rlwe_params.json")
PARAMS_FILENAME = "rlwe_params.json"
ENV_VAR = "LWE_SFTP_RLWE"
_FALLBACK = Params(n=1024, q=12289, k=4)  # only if the packaged file is missing


def _packaged():
    """Load the packaged rlwe_params.json.

    Falls back to _FALLBACK only when the file is missing (a broken install). A file that exists
    but is invalid still raises, so a bad edit is noticed instead of silently ignored.
    """
    try:
        return Params.load(PACKAGED_PARAMS)
    except FileNotFoundError:
        return _FALLBACK


def resolve_params(override=None, config_dir=None):
    """Pick the parameter set. The first source that is set wins:
    1. override (e.g. a --rlwe n,q,k command-line flag); a Params is used as-is, text is parsed
    2. $LWE_SFTP_RLWE
    3. <config_dir>/rlwe_params.json (client: ~/.lwe-sftp, server: its config dir)
    4. the packaged lwe_sftp/rlwe_params.json (DEFAULT)
    Both peers must end up with the same set; the handshake rejects a mismatch by name.
    """
    if override:
        return override if isinstance(override, Params) else Params.parse(override)
    if os.environ.get(ENV_VAR):
        return Params.parse(os.environ[ENV_VAR])
    if config_dir:
        path = os.path.join(config_dir, PARAMS_FILENAME)
        if os.path.exists(path):
            return Params.load(path)
    return DEFAULT


DEFAULT = _packaged()

# backwards-compatible module constants for the default parameter set
N, Q, K_ERROR = DEFAULT.n, DEFAULT.q, DEFAULT.k
Q_HALF = DEFAULT.q_half
POLY_BYTES = DEFAULT.poly_bytes
HINT_BYTES = DEFAULT.hint_bytes


def _center(v, P):
    """Map coefficients to the centered range [-(q-1)/2, (q-1)/2].

    Reduces mod q into [0, q) first, then subtracts q from everything above q_half. Returns int64.
    """
    v = np.asarray(v, dtype=np.int64) % P.q
    return np.where(v > P.q_half, v - P.q, v)


def sample_uniform(P=DEFAULT):
    """Uniform polynomial in R_q via rejection sampling (no modulo bias).

    Draws 32-bit words from os.urandom, masks each to q's bit length and keeps only values below q
    (more than half of them survive). Repeats until there are n, then truncates to exactly n.
    """
    mask = (1 << P.q.bit_length()) - 1
    parts, have = [], 0
    while have < P.n:
        raw = np.frombuffer(os.urandom(4 * P.n), dtype="<u4") & mask
        good = raw[raw < P.q].astype(np.int64)
        parts.append(good)
        have += good.size
    return np.concatenate(parts)[: P.n]


def sample_cbd(P=DEFAULT):
    """Small polynomial from the centered binomial distribution B_k (values in [-k, k], variance k/2).

    Draws 2*k*n random bits, arranges them as n pairs of k-bit groups, and returns
    popcount(first group) - popcount(second group) for each coefficient.
    """
    nbits = 2 * P.k * P.n
    bits = np.unpackbits(np.frombuffer(os.urandom((nbits + 7) // 8), dtype=np.uint8))[:nbits]
    bits = bits.reshape(P.n, 2, P.k).astype(np.int64)
    return bits[:, 0].sum(axis=1) - bits[:, 1].sum(axis=1)


# --- multiplication ------------------------------------------------------------
#
# Negacyclic product via a zero-padded real FFT of length 2n (linear convolution), then fold
# x^n = -1. O(n log n) instead of np.convolve's O(n^2). Inputs are centered first so every
# coefficient of the exact product is < n*(q/2)*k in magnitude, which Params bounds well
# under float64's 2^53 so rounding recovers it exactly.

def _spectrum(a, P):
    """Forward transform for poly_mul.

    Centers the coefficients (which keeps the exact product small), converts to float64 and takes
    the real FFT zero-padded to length 2n, so a pointwise product corresponds to a linear
    convolution rather than a cyclic one.
    """
    return np.fft.rfft(_center(a, P).astype(np.float64), 2 * P.n)


def _from_spectrum(spec, P):
    """Turn a product of two spectra back into a ring element.

    Inverse real FFT to the length-2n linear convolution, round to the nearest integer (exact under
    the Params bound), then fold the upper half onto the lower with a minus sign (x^n = -1) and
    reduce mod q.
    """
    full = np.rint(np.fft.irfft(spec, 2 * P.n)).astype(np.int64)
    return (full[: P.n] - full[P.n:]) % P.q


def poly_mul(a, b, P=DEFAULT):
    """a * b mod (x^n + 1, q) in O(n log n).

    Multiplies the two zero-padded spectra pointwise and folds the result back into the ring. Exact
    for every set Params accepts; the tests check it against poly_mul_schoolbook, including the
    worst case for rounding.
    """
    return _from_spectrum(_spectrum(a, P) * _spectrum(b, P), P)


def poly_mul_schoolbook(a, b, P=DEFAULT):
    """Reference O(n^2) implementation, kept for tests and benchmarks.

    Computes the full linear convolution with np.convolve, subtracts the coefficients of
    x^n..x^(2n-2) from the low half (x^n = -1) and reduces mod q.
    """
    full = np.convolve(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64))
    res = full[: P.n].copy()
    res[: P.n - 1] -= full[P.n:]
    return res % P.q


# --- Ding et al. reconciliation ---------------------------------------------

def signal(v, P=DEFAULT):
    """Sig(v), Ding's HelpRec: 0 if the centered coefficient lies in E = [-q/4, q/4], else 1.

    The responder sends these bits so both sides know which coefficients are near the +-q/2
    wrap-around and must be rotated before Mod2.
    """
    return (np.abs(_center(v, P)) > P.q // 4).astype(np.int64)


def mod2(v, w, P=DEFAULT):
    """Mod2(v, w) = ((v + w*(q-1)/2) mod q, centered) mod 2.

    Adds (q-1)/2 to the coefficients the hint flags, which rotates them away from the wrap point,
    centers the result and takes its parity. Returns a uint8 bit array.

    Centering matters: reducing to [0, q) instead puts the wrap-around exactly where
    Sig steers values, and roughly 4% of exchanges then disagree on some bit.
    """
    shifted = np.asarray(v, dtype=np.int64) + np.asarray(w, dtype=np.int64) * P.q_half
    return (_center(shifted, P) % 2).astype(np.uint8)


# --- protocol roles ---------------------------------------------------------

def initiate(P=DEFAULT):
    """Initiator's first step. Returns (m, secret, public).

    Samples a fresh uniform public polynomial m, a small secret s and error e, and computes the
    public key P_i = m*s + 2e. m and public go in the client hello; secret stays local for finish().
    """
    m = sample_uniform(P)
    s, e = sample_cbd(P), sample_cbd(P)
    return m, s, (poly_mul(m, s, P) + 2 * e) % P.q


def respond(m, initiator_public, P=DEFAULT):
    """Responder's only step. Returns (public, hint, key_bits).

    Samples a secret s and errors e, e', and computes its public key P_r = m*s + 2e and its
    approximation of the shared value v = P_i*s + 2e'. The hint is Sig(v) and the key bits are
    Mod2(v, hint). s is transformed to the FFT domain once and reused for both products.
    """
    s, e, e2 = sample_cbd(P), sample_cbd(P), sample_cbd(P)
    fs = _spectrum(s, P)  # s multiplies twice; transform it once
    public = (_from_spectrum(_spectrum(m, P) * fs, P) + 2 * e) % P.q
    v = (_from_spectrum(_spectrum(initiator_public, P) * fs, P) + 2 * e2) % P.q
    w = signal(v, P)
    return public, w, mod2(v, w, P)


def finish(secret, responder_public, hint, P=DEFAULT):
    """Initiator's final step. Returns key_bits.

    Computes its own approximation v = s_i*P_r + 2e'' (close to the responder's, since both are
    about m*s_i*s_r) and applies the responder's hint with Mod2. The result equals the responder's
    bits except with the tiny failure probability the parameters were chosen for.
    """
    v = (poly_mul(secret, responder_public, P) + 2 * sample_cbd(P)) % P.q
    return mod2(v, hint, P)


# --- wire encoding ----------------------------------------------------------

def encode_poly(p, P=DEFAULT):
    """Serialize a polynomial (coefficients already in [0, q)) as little-endian integers of coef_dtype width."""
    return np.asarray(p, dtype=P.coef_dtype).tobytes()


def decode_poly(data, P=DEFAULT):
    """Parse a polynomial from untrusted bytes.

    Checks the length is exactly poly_bytes and every coefficient is below q, raising ValueError
    otherwise. Returns an int64 array.
    """
    if len(data) != P.poly_bytes:
        raise ValueError("bad R-LWE polynomial length")
    p = np.frombuffer(data, dtype=P.coef_dtype).astype(np.int64)
    if (p >= P.q).any():
        raise ValueError("R-LWE polynomial coefficient out of range")
    return p


def encode_bits(bits):
    """Pack a 0/1 array into bytes, eight bits per byte, most significant bit first."""
    return np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes()


def decode_bits(data, P=DEFAULT):
    """Unpack hint bytes into n 0/1 values.

    Checks the length is exactly hint_bytes, unpacks, and drops the padding bits past n.
    """
    if len(data) != P.hint_bytes:
        raise ValueError("bad R-LWE hint length")
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))[: P.n].astype(np.int64)
