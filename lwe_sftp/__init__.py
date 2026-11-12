"""LWE-SFTP: file transfer over a hybrid post-quantum (R-LWE + ML-KEM + X25519) channel.

This package root holds the constants every module shares: the version, the protocol
identifier (sent in the handshake and mixed into every signature and key-derivation context, so
messages from one protocol version can't be reused in another) and the default TCP port.

Modules, bottom layer first:
    rlwe       Ring-LWE key exchange (Ding, Xie & Lin) and its parameter sets
    keys       Ed25519 key files, known_hosts, scrypt password hashes
    transport  hybrid handshake and the ChaCha20-Poly1305 record layer it produces
    server     lsftpd: authenticates users and serves file operations inside a root directory
    client     client library: host-key check, login, file operations and transfers
    cli        the lscp / lsftp / lsftpd command-line front ends
"""

__version__ = "0.2.0"

PROTOCOL = "LWE-SFTP/1"
DEFAULT_PORT = 4422
