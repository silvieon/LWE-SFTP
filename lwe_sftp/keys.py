"""Key files, known_hosts, authorized keys and password hashes (OpenSSH-compatible formats)."""

import base64
import hashlib
import hmac
import os

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from . import PROTOCOL

AUTH_CTX = PROTOCOL.encode() + b" userauth\x00"


def _read_umask():
    old = os.umask(0)
    os.umask(old)
    return old


# process umask, read once at import (os.umask can only be read by setting it, which races threads)
UMASK = _read_umask()


class KeyFileError(Exception):
    pass


# --- Ed25519 keys -----------------------------------------------------------

def generate_private_key(path):
    key = Ed25519PrivateKey.generate()
    data = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.OpenSSH, serialization.NoEncryption()
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return key


def load_private_key(path, passphrase_cb=None):
    """Load an Ed25519 key in OpenSSH or PEM format (so ~/.ssh/id_ed25519 works as-is)."""
    with open(path, "rb") as f:
        data = f.read()
    loader = serialization.load_ssh_private_key if b"OPENSSH PRIVATE KEY" in data else serialization.load_pem_private_key
    try:
        key = loader(data, password=None)
    except TypeError:
        # encrypted key; OpenSSH-format decryption needs the 'bcrypt' package installed
        if passphrase_cb is None:
            raise KeyFileError(f"{path} is passphrase-protected") from None
        key = loader(data, password=passphrase_cb(path).encode())
    if not isinstance(key, Ed25519PrivateKey):
        raise KeyFileError(f"{path}: only Ed25519 keys are supported")
    return key


def public_to_openssh(pub):
    return pub.public_bytes(serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH).decode()


def parse_openssh_public(line):
    """Parse 'ssh-ed25519 AAAA... [comment]'. Returns None for anything else."""
    parts = line.strip().split()
    if len(parts) < 2 or parts[0] != "ssh-ed25519":
        return None
    try:
        key = serialization.load_ssh_public_key(f"{parts[0]} {parts[1]}".encode())
    except (ValueError, TypeError):
        return None
    return key if isinstance(key, Ed25519PublicKey) else None


def fingerprint(pub):
    """Same format as `ssh-keygen -lf`."""
    blob = base64.b64decode(public_to_openssh(pub).split()[1])
    return "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=")


def raw(pub):
    return pub.public_bytes_raw()


def auth_payload(session_id, user):
    return AUTH_CTX + session_id + user.encode()


# --- known_hosts ------------------------------------------------------------

def host_label(host, port):
    return f"[{host}]:{port}"


class KnownHosts:
    """'[host]:port ssh-ed25519 AAAA...' lines; trust-on-first-use like OpenSSH."""

    def __init__(self, path):
        self.path = path

    def lookup(self, label):
        """Returns the list of keys recorded for label."""
        keys = []
        try:
            with open(self.path) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 3 and parts[0] == label:
                        key = parse_openssh_public(" ".join(parts[1:3]))
                        if key is not None:
                            keys.append(key)
        except FileNotFoundError:
            pass
        return keys

    def add(self, label, pub):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "a") as f:
            f.write(f"{label} {public_to_openssh(pub)}\n")


# --- passwords --------------------------------------------------------------

_SCRYPT = dict(n=2 ** 15, r=8, p=1, maxmem=64 * 1024 * 1024)


def hash_password(password):
    salt = os.urandom(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, dklen=32, **_SCRYPT)
    return "scrypt${}${}${}${}${}".format(
        _SCRYPT["n"], _SCRYPT["r"], _SCRYPT["p"], base64.b64encode(salt).decode(), base64.b64encode(dk).decode()
    )


_DUMMY_HASH = None


def verify_password(stored, password):
    """Constant-ish time; hashes against a dummy record when the user/password is unknown."""
    global _DUMMY_HASH
    if not stored:
        if _DUMMY_HASH is None:
            _DUMMY_HASH = hash_password("dummy")
        stored, password, real = _DUMMY_HASH, password or "", False
    else:
        real = True
    try:
        scheme, n, r, p, salt, dk = stored.split("$")
        if scheme != "scrypt":
            return False
        dk = base64.b64decode(dk)
        test = hashlib.scrypt(
            password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p),
            dklen=len(dk), maxmem=_SCRYPT["maxmem"],
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(test, dk) and real
