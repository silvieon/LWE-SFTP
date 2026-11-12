"""Key files, known_hosts, authorized keys and password hashes (OpenSSH-compatible formats).

Used by both ends. The server loads its Ed25519 host key and checks users' public keys and
password hashes from users.json. The client loads the user's Ed25519 identity, signs the login
payload, and pins server host keys in known_hosts. Keys, fingerprints and known_hosts lines use
OpenSSH formats, so ssh-keygen output works here and vice versa. Passwords are hashed with scrypt.
"""

import base64
import hashlib
import hmac
import os

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from . import PROTOCOL

AUTH_CTX = PROTOCOL.encode() + b" userauth\x00"


def _read_umask():
    """Return the process umask.

    os.umask can only be read by setting it, so this sets 0 and immediately restores the old value.
    It runs once at import time, before any server threads exist.
    """
    old = os.umask(0)
    os.umask(old)
    return old


# process umask, read once at import (os.umask can only be read by setting it, which races threads)
UMASK = _read_umask()


class KeyFileError(Exception):
    """A key file exists but can't be used: passphrase-protected with no prompt available, or not Ed25519."""


# --- Ed25519 keys -----------------------------------------------------------

def generate_private_key(path):
    """Create an Ed25519 private key and save it to path, unencrypted, in OpenSSH format.

    Creates the parent directory if needed. The file is opened with O_EXCL and mode 0600, so an
    existing key is never overwritten and the new one is never readable by others. Returns the key.
    """
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
    """Load an Ed25519 key in OpenSSH or PEM format (so ~/.ssh/id_ed25519 works as-is).

    Picks the loader from the file header and first tries without a password. If the key is
    encrypted, asks passphrase_cb(path) for a passphrase, or raises KeyFileError when there is no
    callback. Any key type other than Ed25519 raises KeyFileError.
    """
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
    """Format a public key as an OpenSSH line, 'ssh-ed25519 AAAA...', without a comment."""
    return pub.public_bytes(serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH).decode()


def parse_openssh_public(line):
    """Parse 'ssh-ed25519 AAAA... [comment]'. Returns None for anything else.

    Only the type and base64 fields are parsed; the comment is ignored. Blank lines, comments,
    other key types and corrupt data all give None, so callers can map this over an
    authorized_keys-style file and drop the Nones.
    """
    parts = line.strip().split()
    if len(parts) < 2 or parts[0] != "ssh-ed25519":
        return None
    try:
        key = serialization.load_ssh_public_key(f"{parts[0]} {parts[1]}".encode())
    except (ValueError, TypeError):
        return None
    return key if isinstance(key, Ed25519PublicKey) else None


def fingerprint(pub):
    """Same format as `ssh-keygen -lf`: 'SHA256:' plus the unpadded base64 SHA-256 of the key blob."""
    blob = base64.b64decode(public_to_openssh(pub).split()[1])
    return "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=")


def raw(pub):
    """The 32 raw public-key bytes, used to compare keys however they were encoded or commented."""
    return pub.public_bytes_raw()


def auth_payload(session_id, user):
    """The bytes a client signs to log in: a fixed context string, the session id and the user name.

    The session id is the handshake transcript hash, so a signature is only valid on the connection
    it was made for. The context string keeps it distinct from the server's handshake signature.
    """
    return AUTH_CTX + session_id + user.encode()


# --- known_hosts ------------------------------------------------------------

def host_label(host, port):
    """known_hosts label for a server: '[host]:port', the form OpenSSH uses for non-default ports."""
    return f"[{host}]:{port}"


class KnownHosts:
    """'[host]:port ssh-ed25519 AAAA...' lines; trust-on-first-use like OpenSSH.

    The file is read again on every lookup and only ever appended to.
    """

    def __init__(self, path):
        """Remember the file path; nothing is read until lookup()."""
        self.path = path

    def lookup(self, label):
        """Returns the list of keys recorded for label.

        Scans the file line by line, keeping lines whose first field is label and whose next two fields
        parse as an ssh-ed25519 key. A missing file means no keys.
        """
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
        """Append a 'label ssh-ed25519 AAAA...' line, creating the parent directory if needed."""
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "a") as f:
            f.write(f"{label} {public_to_openssh(pub)}\n")


# --- passwords --------------------------------------------------------------

_SCRYPT = dict(n=2 ** 15, r=8, p=1, maxmem=64 * 1024 * 1024)


def hash_password(password):
    """Hash a password for users.json.

    Uses scrypt (N=2^15, r=8, p=1) with a random 16-byte salt and returns
    'scrypt$N$r$p$salt$hash' with base64 fields. The cost parameters are stored with the hash, so
    they can be raised later without invalidating existing records.
    """
    salt = os.urandom(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, dklen=32, **_SCRYPT)
    return "scrypt${}${}${}${}${}".format(
        _SCRYPT["n"], _SCRYPT["r"], _SCRYPT["p"], base64.b64encode(salt).decode(), base64.b64encode(dk).decode()
    )


_DUMMY_HASH = None


def verify_password(stored, password):
    """Check a password against a stored record. Constant-ish time.

    Splits the record, re-derives the hash with the record's own parameters and salt, and compares
    with hmac.compare_digest. With no stored record (unknown user, or no password set) it hashes
    against a lazily created dummy record instead, so the reply takes as long as a real check, and
    returns False. A malformed record also returns False.
    """
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
