"""Transport: the hybrid handshake and the encrypted record layer it produces.

Key exchange: R-LWE (Ding et al.) + ML-KEM-768 + X25519, server-authenticated with Ed25519.

One round trip:
    C -> S  hello   {proto, rlwe (parameter set name), x25519, mlkem_ek, rlwe_m, rlwe_p}
    S -> C  reply   {proto, x25519, mlkem_ct, rlwe_p, rlwe_hint, host_key}
    S -> C  sig     Ed25519(host_key, SIG_CTX || H)          H = SHA-256 transcript of hello, reply

    traffic keys = HKDF-SHA256(ikm = x25519_ss || mlkem_ss || rlwe_bits, salt = H)

The session stays confidential as long as ANY of the three exchanges holds, so the
experimental R-LWE parameters can never make things weaker than X25519 + ML-KEM alone.

After the handshake every message is a ChaCha20-Poly1305 record (see SecureChannel).
"""

import base64
import binascii
import hashlib
import json
import struct

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import mlkem
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from . import PROTOCOL, rlwe

# ============================================================================
# Framing and record layer
# ============================================================================

# record kinds
MSG = 1    # JSON request/response
DATA = 2   # raw file bytes
EOD = 3    # end of a data stream; payload is a JSON status

MAX_RECORD = 1 << 20
CHUNK = 64 * 1024
_LEN = struct.Struct(">I")
_TAG = 16


class ProtocolError(Exception):
    pass


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 16))
        if not chunk:
            raise ConnectionError("connection closed by peer")
        buf += chunk
    return bytes(buf)


def send_frame(sock, data):
    sock.sendall(_LEN.pack(len(data)) + data)


def recv_frame(sock, limit=MAX_RECORD):
    (n,) = _LEN.unpack(recv_exact(sock, 4))
    if n > limit:
        raise ProtocolError(f"frame too large ({n} bytes)")
    return recv_exact(sock, n)


def _nonce(seq):
    return b"\x00" * 4 + struct.pack(">Q", seq)


class SecureChannel:
    """ChaCha20-Poly1305 records with per-direction keys and implicit sequence-number nonces.

    The length header is authenticated as associated data; reordering, replay, truncation
    mid-record and tampering all surface as ProtocolError.
    """

    def __init__(self, sock, send_key, recv_key, session_id):
        self.sock = sock
        self.session_id = session_id
        self._tx = ChaCha20Poly1305(send_key)
        self._rx = ChaCha20Poly1305(recv_key)
        self._tx_seq = 0
        self._rx_seq = 0

    def send(self, kind, payload=b""):
        pt = bytes([kind]) + payload
        header = _LEN.pack(len(pt) + _TAG)
        ct = self._tx.encrypt(_nonce(self._tx_seq), pt, header)
        self._tx_seq += 1
        self.sock.sendall(header + ct)

    def recv(self):
        header = recv_exact(self.sock, 4)
        (n,) = _LEN.unpack(header)
        if n < _TAG + 1 or n > MAX_RECORD + _TAG + 1:
            raise ProtocolError(f"bad record length {n}")
        ct = recv_exact(self.sock, n)
        try:
            pt = self._rx.decrypt(_nonce(self._rx_seq), ct, header)
        except InvalidTag:
            raise ProtocolError("record authentication failed (tampering or key mismatch)") from None
        self._rx_seq += 1
        return pt[0], pt[1:]

    # convenience wrappers

    def send_msg(self, obj):
        self.send(MSG, json.dumps(obj, separators=(",", ":")).encode())

    def recv_msg(self):
        kind, payload = self.recv()
        if kind != MSG:
            raise ProtocolError(f"expected message record, got kind {kind}")
        return _load_obj(payload)

    def send_data(self, chunk):
        self.send(DATA, chunk)

    def send_eod(self, error=None):
        self.send(EOD, json.dumps({"ok": error is None, "error": error}).encode())

    def recv_chunk(self):
        """Returns (bytes, None) for a data record or (None, status) at end of stream."""
        kind, payload = self.recv()
        if kind == DATA:
            return payload, None
        if kind == EOD:
            return None, _load_obj(payload)
        raise ProtocolError(f"expected data record, got kind {kind}")


def _load_obj(payload):
    try:
        obj = json.loads(payload)
    except ValueError:
        raise ProtocolError("malformed message") from None
    if not isinstance(obj, dict):
        raise ProtocolError("malformed message")
    return obj


# ============================================================================
# Handshake
# ============================================================================

SIG_CTX = PROTOCOL.encode() + b" server signature\x00"
KDF_INFO = PROTOCOL.encode() + b" traffic keys"
_HANDSHAKE_LIMIT = 64 * 1024


class HandshakeError(Exception):
    pass


class _Transcript:
    def __init__(self):
        self._h = hashlib.sha256(PROTOCOL.encode() + b" transcript\x00")

    def add(self, frame):
        self._h.update(len(frame).to_bytes(4, "big") + frame)

    def digest(self):
        return self._h.digest()


def _b64e(b):
    return base64.b64encode(b).decode()


def _b64d(s):
    return base64.b64decode(s, validate=True)


def _derive(sock, x_ss, kem_ss, rlwe_bits, h, is_client):
    okm = HKDF(algorithm=hashes.SHA256(), length=64, salt=h, info=KDF_INFO).derive(
        x_ss + kem_ss + rlwe.encode_bits(rlwe_bits)
    )
    c2s, s2c = okm[:32], okm[32:]
    if is_client:
        return SecureChannel(sock, c2s, s2c, h)
    return SecureChannel(sock, s2c, c2s, h)


def client_handshake(sock, params=rlwe.DEFAULT):
    """Returns (channel, server_host_public_key). Caller MUST check the host key before trusting it."""
    x_priv = X25519PrivateKey.generate()
    kem_priv = mlkem.MLKEM768PrivateKey.generate()
    P = params
    m, r_secret, r_public = rlwe.initiate(P)

    hello = json.dumps({
        "proto": PROTOCOL,
        "rlwe": P.name,
        "x25519": _b64e(x_priv.public_key().public_bytes_raw()),
        "mlkem_ek": _b64e(kem_priv.public_key().public_bytes_raw()),
        "rlwe_m": _b64e(rlwe.encode_poly(m, P)),
        "rlwe_p": _b64e(rlwe.encode_poly(r_public, P)),
    }).encode()
    t = _Transcript()
    t.add(hello)
    send_frame(sock, hello)

    reply_raw = recv_frame(sock, _HANDSHAKE_LIMIT)
    try:
        reply = json.loads(reply_raw)
    except ValueError:
        raise HandshakeError("malformed server handshake") from None
    if isinstance(reply, dict) and "error" in reply:
        # a refusal is a single frame; don't wait for a signature that will never come
        raise HandshakeError(f"server refused handshake: {reply['error']}")
    sig_raw = recv_frame(sock, _HANDSHAKE_LIMIT)
    t.add(reply_raw)
    h = t.digest()

    try:
        if reply.get("proto") != PROTOCOL:
            raise HandshakeError(f"unsupported server protocol {reply.get('proto')!r}")
        host_key = Ed25519PublicKey.from_public_bytes(_b64d(reply["host_key"]))
        host_key.verify(_b64d(json.loads(sig_raw)["sig"]), SIG_CTX + h)

        x_ss = x_priv.exchange(X25519PublicKey.from_public_bytes(_b64d(reply["x25519"])))
        kem_ss = kem_priv.decapsulate(_b64d(reply["mlkem_ct"]))
        bits = rlwe.finish(
            r_secret, rlwe.decode_poly(_b64d(reply["rlwe_p"]), P), rlwe.decode_bits(_b64d(reply["rlwe_hint"]), P), P
        )
    except InvalidSignature:
        raise HandshakeError("server host-key signature is invalid") from None
    except (ValueError, KeyError, TypeError, binascii.Error) as e:
        raise HandshakeError(f"malformed server handshake: {e}") from None

    return _derive(sock, x_ss, kem_ss, bits, h, is_client=True), host_key


def server_handshake(sock, host_private_key, params=rlwe.DEFAULT):
    hello_raw = recv_frame(sock, _HANDSHAKE_LIMIT)
    try:
        hello = json.loads(hello_raw)
        if hello.get("proto") != PROTOCOL:
            send_frame(sock, json.dumps({"error": f"unsupported protocol, server speaks {PROTOCOL}"}).encode())
            raise HandshakeError(f"client spoke {hello.get('proto')!r}")
        P = params
        if hello.get("rlwe") != P.name:
            send_frame(sock, json.dumps({"error": f"R-LWE parameter mismatch, server uses {P.name}"}).encode())
            raise HandshakeError(f"client R-LWE parameters {hello.get('rlwe')!r} != {P.name}")

        x_priv = X25519PrivateKey.generate()
        x_ss = x_priv.exchange(X25519PublicKey.from_public_bytes(_b64d(hello["x25519"])))
        kem_ss, kem_ct = mlkem.MLKEM768PublicKey.from_public_bytes(_b64d(hello["mlkem_ek"])).encapsulate()
        r_public, hint, bits = rlwe.respond(
            rlwe.decode_poly(_b64d(hello["rlwe_m"]), P), rlwe.decode_poly(_b64d(hello["rlwe_p"]), P), P
        )
    except (ValueError, KeyError, TypeError, AttributeError, binascii.Error) as e:
        raise HandshakeError(f"malformed client hello: {e}") from None

    reply = json.dumps({
        "proto": PROTOCOL,
        "x25519": _b64e(x_priv.public_key().public_bytes_raw()),
        "mlkem_ct": _b64e(kem_ct),
        "rlwe_p": _b64e(rlwe.encode_poly(r_public, P)),
        "rlwe_hint": _b64e(rlwe.encode_bits(hint)),
        "host_key": _b64e(host_private_key.public_key().public_bytes_raw()),
    }).encode()
    t = _Transcript()
    t.add(hello_raw)
    t.add(reply)
    h = t.digest()
    send_frame(sock, reply)
    send_frame(sock, json.dumps({"sig": _b64e(host_private_key.sign(SIG_CTX + h))}).encode())

    return _derive(sock, x_ss, kem_ss, bits, h, is_client=False)
