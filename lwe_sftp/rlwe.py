"""Ring-LWE key exchange after Ding, Xie & Lin (IACR ePrint 2012/688).

Arithmetic is in R_q = Z_q[x] / (x^n + 1). All randomness comes from os.urandom.

Flow (the initiator picks the public polynomial m):
    initiator:  m <- uniform, s_i, P_i = m*s_i + 2e_i              -> (m, P_i)
    responder:  s_r, P_r = m*s_r + 2e_r
                v_r = P_i*s_r + 2e'_r,  w = Sig(v_r),  k = Mod2(v_r, w)  -> (P_r, w)
    initiator:  v_i = P_r*s_i + 2e'_i,  k = Mod2(v_i, w)

Parameters are a Params value so they can be benchmarked/swept (see tools/bench.py); DEFAULT is what
the protocol uses, and both peers must agree on it (it's part of the handshake).
"""

import json
import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Params:
    n: int   # ring degree, power of two
    q: int   # odd modulus (FFT multiplication means it need not be NTT-friendly)
    k: int   # centered-binomial parameter for secrets/errors: B_k, variance k/2

    def __post_init__(self):
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
        return f"n={self.n},q={self.q},k={self.k}"

    @property
    def q_half(self):
        return (self.q - 1) // 2

    @property
    def coef_dtype(self):
        return "<u2" if self.q <= 1 << 16 else "<u4"

    @property
    def poly_bytes(self):
        return self.n * np.dtype(self.coef_dtype).itemsize

    @property
    def hint_bytes(self):
        return (self.n + 7) // 8

    @classmethod
    def parse(cls, text):
        """'1024,12289,4' or 'n=1024,q=12289,k=4' (the form Params.name produces)."""
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
        """Read a params file as written by tools/bench.py sweep --write ({"n":..,"q":..,"k":.., ...metadata})."""
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
    try:
        return Params.load(PACKAGED_PARAMS)
    except FileNotFoundError:
        return _FALLBACK


def resolve_params(override=None, config_dir=None):
    """Pick the parameter set, highest priority first:
    1. override (e.g. a --rlwe n,q,k command-line flag)
    2. $LWE_SFTP_RLWE
    3. <config_dir>/rlwe_params.json (client: ~/.lwe-sftp, server: its config dir)
    4. the packaged lwe_sftp/rlwe_params.json
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
    """Map coefficients to the centered range [-(q-1)/2, (q-1)/2]."""
    v = np.asarray(v, dtype=np.int64) % P.q
    return np.where(v > P.q_half, v - P.q, v)


def sample_uniform(P=DEFAULT):
    """Uniform polynomial in R_q via rejection sampling (no modulo bias)."""
    mask = (1 << P.q.bit_length()) - 1
    parts, have = [], 0
    while have < P.n:
        raw = np.frombuffer(os.urandom(4 * P.n), dtype="<u4") & mask
        good = raw[raw < P.q].astype(np.int64)
        parts.append(good)
        have += good.size
    return np.concatenate(parts)[: P.n]


def sample_cbd(P=DEFAULT):
    """Small polynomial from the centered binomial distribution B_k."""
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
    return np.fft.rfft(_center(a, P).astype(np.float64), 2 * P.n)


def _from_spectrum(spec, P):
    full = np.rint(np.fft.irfft(spec, 2 * P.n)).astype(np.int64)
    return (full[: P.n] - full[P.n:]) % P.q


def poly_mul(a, b, P=DEFAULT):
    """a * b mod (x^n + 1, q)."""
    return _from_spectrum(_spectrum(a, P) * _spectrum(b, P), P)


def poly_mul_schoolbook(a, b, P=DEFAULT):
    """Reference O(n^2) implementation, kept for tests and benchmarks."""
    full = np.convolve(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64))
    res = full[: P.n].copy()
    res[: P.n - 1] -= full[P.n:]
    return res % P.q


# --- Ding et al. reconciliation ---------------------------------------------

def signal(v, P=DEFAULT):
    """Sig(v): 0 if the centered coefficient lies in E = [-q/4, q/4], else 1."""
    return (np.abs(_center(v, P)) > P.q // 4).astype(np.int64)


def mod2(v, w, P=DEFAULT):
    """Mod2(v, w) = ((v + w*(q-1)/2) mod q, centered) mod 2.

    Centering matters: reducing to [0, q) instead puts the wrap-around exactly where
    Sig steers values, and roughly 4% of exchanges then disagree on some bit.
    """
    shifted = np.asarray(v, dtype=np.int64) + np.asarray(w, dtype=np.int64) * P.q_half
    return (_center(shifted, P) % 2).astype(np.uint8)


# --- protocol roles ---------------------------------------------------------

def initiate(P=DEFAULT):
    """Returns (m, secret, public)."""
    m = sample_uniform(P)
    s, e = sample_cbd(P), sample_cbd(P)
    return m, s, (poly_mul(m, s, P) + 2 * e) % P.q


def respond(m, initiator_public, P=DEFAULT):
    """Returns (public, hint, key_bits)."""
    s, e, e2 = sample_cbd(P), sample_cbd(P), sample_cbd(P)
    fs = _spectrum(s, P)  # s multiplies twice; transform it once
    public = (_from_spectrum(_spectrum(m, P) * fs, P) + 2 * e) % P.q
    v = (_from_spectrum(_spectrum(initiator_public, P) * fs, P) + 2 * e2) % P.q
    w = signal(v, P)
    return public, w, mod2(v, w, P)


def finish(secret, responder_public, hint, P=DEFAULT):
    """Initiator's final step. Returns key_bits."""
    v = (poly_mul(secret, responder_public, P) + 2 * sample_cbd(P)) % P.q
    return mod2(v, hint, P)


# --- wire encoding ----------------------------------------------------------

def encode_poly(p, P=DEFAULT):
    return np.asarray(p, dtype=P.coef_dtype).tobytes()


def decode_poly(data, P=DEFAULT):
    if len(data) != P.poly_bytes:
        raise ValueError("bad R-LWE polynomial length")
    p = np.frombuffer(data, dtype=P.coef_dtype).astype(np.int64)
    if (p >= P.q).any():
        raise ValueError("R-LWE polynomial coefficient out of range")
    return p


def encode_bits(bits):
    return np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes()


def decode_bits(data, P=DEFAULT):
    if len(data) != P.hint_bytes:
        raise ValueError("bad R-LWE hint length")
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))[: P.n].astype(np.int64)
