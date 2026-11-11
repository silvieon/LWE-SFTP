"""lsftpd: the LWE-SFTP server. One TCP connection per session, every user confined to a root dir."""

import base64
import binascii
import json
import logging
import os
import posixpath
import socket
import socketserver
import stat
import tempfile
import time

from cryptography.exceptions import InvalidSignature

from . import keys, rlwe
from .transport import CHUNK, HandshakeError, ProtocolError, server_handshake

log = logging.getLogger("lsftpd")

MAX_AUTH_ATTEMPTS = 6
PREAUTH_TIMEOUT = 30
IDLE_TIMEOUT = 3600
LIST_BATCH = 2000


def default_config_dir():
    if os.environ.get("LWE_SFTPD_DIR"):
        return os.environ["LWE_SFTPD_DIR"]
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return "/etc/lwe-sftpd"
    return os.path.expanduser("~/.config/lwe-sftpd")


class Config:
    """<dir>/host_ed25519 and <dir>/users.json. users.json is re-read on every login."""

    def __init__(self, directory):
        self.dir = directory
        self.host_key_path = os.path.join(directory, "host_ed25519")
        self.users_path = os.path.join(directory, "users.json")

    def load_host_key(self, create=True):
        if create and not os.path.exists(self.host_key_path):
            os.makedirs(self.dir, mode=0o700, exist_ok=True)
            key = keys.generate_private_key(self.host_key_path)
            log.info("generated host key %s (%s)", self.host_key_path, keys.fingerprint(key.public_key()))
        else:
            key = keys.load_private_key(self.host_key_path)
        pub_path = self.host_key_path + ".pub"
        if create and not os.path.exists(pub_path):
            with open(pub_path, "w") as f:
                f.write(keys.public_to_openssh(key.public_key()) + "\n")
        return key

    def load_users(self):
        try:
            with open(self.users_path) as f:
                users = json.load(f)
        except FileNotFoundError:
            return {}
        return users if isinstance(users, dict) else {}

    def save_users(self, users):
        os.makedirs(self.dir, mode=0o700, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=".users-")
        with os.fdopen(fd, "w") as f:
            json.dump(users, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, self.users_path)


class FsError(Exception):
    pass


def _attrs(st):
    if stat.S_ISDIR(st.st_mode):
        kind = "dir"
    elif stat.S_ISREG(st.st_mode):
        kind = "file"
    else:
        kind = "other"
    return {"type": kind, "size": st.st_size, "mode": stat.S_IMODE(st.st_mode), "mtime": st.st_mtime}


def _opt_int(value, name):
    if value is None or (isinstance(value, (int, float)) and not isinstance(value, bool)):
        return value
    raise FsError(f"invalid {name}")


class Session:
    OPS = ("stat", "list", "get", "put", "mkdir", "rmdir", "remove", "rename")

    def __init__(self, sock, addr, config, host_key, rlwe_params=rlwe.DEFAULT):
        self.sock = sock
        self.peer = f"{addr[0]}:{addr[1]}"
        self.config = config
        self.host_key = host_key
        self.rlwe_params = rlwe_params
        self.chan = None
        self.user = None
        self.root = None

    def run(self):
        self.sock.settimeout(PREAUTH_TIMEOUT)
        try:
            self.chan = server_handshake(self.sock, self.host_key, self.rlwe_params)
            if not self._authenticate():
                return
            self.sock.settimeout(IDLE_TIMEOUT)
            while True:
                self._dispatch(self.chan.recv_msg())
        except (HandshakeError, ProtocolError) as e:
            log.warning("%s: %s", self.peer, e)
        except (ConnectionError, TimeoutError, OSError) as e:
            log.info("%s: disconnected (%s)", self.peer, e)
        finally:
            try:
                self.sock.close()
            except OSError:
                pass

    # --- authentication ------------------------------------------------------

    def _authenticate(self):
        for _ in range(MAX_AUTH_ATTEMPTS):
            req = self.chan.recv_msg()
            if req.get("op") != "auth":
                self.chan.send_msg({"ok": False, "error": "authentication required"})
                continue
            user = req.get("user") if isinstance(req.get("user"), str) else ""
            record = self.config.load_users().get(user) if user else None
            method = req.get("method")
            ok = False
            if method == "publickey":
                ok = self._check_publickey(user, record, req)
            elif method == "password":
                pw = req.get("password")
                ok = isinstance(pw, str) and keys.verify_password(record.get("password") if record else None, pw)
                if not ok:
                    time.sleep(1)
            if ok:
                root = os.path.realpath(os.path.expanduser(record.get("root", "")))
                if not record.get("root") or not os.path.isdir(root):
                    log.error("%s: user %s has no usable root directory %r", self.peer, user, record.get("root"))
                    self.chan.send_msg({"ok": False, "error": "home directory unavailable"})
                    return False
                self.user, self.root = user, root
                self.chan.send_msg({"ok": True, "user": user})
                log.info("%s: %s logged in via %s", self.peer, user, method)
                return True
            log.warning("%s: failed %s auth for %r", self.peer, method, user)
            self.chan.send_msg({"ok": False, "error": "authentication failed"})
        return False

    def _check_publickey(self, user, record, req):
        pub = keys.parse_openssh_public(str(req.get("key", "")))
        if pub is None or record is None:
            return False
        allowed = {keys.raw(k) for k in map(keys.parse_openssh_public, record.get("authorized_keys", [])) if k}
        if keys.raw(pub) not in allowed:
            return False
        try:
            pub.verify(base64.b64decode(req.get("sig", ""), validate=True), keys.auth_payload(self.chan.session_id, user))
        except (InvalidSignature, ValueError, TypeError, binascii.Error):
            return False
        return True

    # --- paths ---------------------------------------------------------------

    def _inside(self, real):
        return self.root == "/" or real == self.root or real.startswith(self.root + os.sep)

    def _resolve(self, vpath, follow=True):
        """Map a client path ('/' = user root) to a real path, refusing anything that escapes the root.

        follow=False resolves only the parent, so the last component may be a symlink that is
        itself removed/renamed/replaced rather than its target.
        """
        if not isinstance(vpath, str) or "\x00" in vpath:
            raise FsError("invalid path")
        rel = posixpath.normpath("/" + vpath).lstrip("/")
        if follow:
            real = os.path.realpath(os.path.join(self.root, rel))
        else:
            if not rel:
                raise FsError("operation not permitted on the root directory")
            parent, name = posixpath.split(rel)
            real = os.path.join(os.path.realpath(os.path.join(self.root, parent)), name)
        if not self._inside(real if follow else os.path.dirname(real)):
            raise FsError("permission denied")
        return real

    # --- dispatch ------------------------------------------------------------

    def _dispatch(self, req):
        op = req.get("op")
        if op not in self.OPS:
            self.chan.send_msg({"ok": False, "error": f"unknown operation {op!r}"})
            return
        try:
            getattr(self, "op_" + op)(req)
        except FsError as e:
            self.chan.send_msg({"ok": False, "error": str(e)})
        except OSError as e:
            if e.strerror is None:  # socket-level failure
                raise
            self.chan.send_msg({"ok": False, "error": e.strerror})

    def op_stat(self, req):
        self.chan.send_msg({"ok": True, **_attrs(os.stat(self._resolve(req.get("path"))))})

    def op_list(self, req):
        path = self._resolve(req.get("path"))
        batch = []
        with os.scandir(path) as it:
            for entry in it:
                try:
                    st = entry.stat()
                except OSError:
                    st = entry.stat(follow_symlinks=False)
                batch.append({"name": entry.name, **_attrs(st)})
                if len(batch) >= LIST_BATCH:
                    self.chan.send_msg({"ok": True, "entries": batch, "more": True})
                    batch = []
        self.chan.send_msg({"ok": True, "entries": batch, "more": False})

    def op_get(self, req):
        path = self._resolve(req.get("path"))
        with open(path, "rb") as f:
            st = os.fstat(f.fileno())
            if not stat.S_ISREG(st.st_mode):
                raise FsError("not a regular file")
            self.chan.send_msg({"ok": True, **_attrs(st)})
            error = None
            try:
                while chunk := f.read(CHUNK):
                    self.chan.send_data(chunk)
            except OSError as e:
                if e.strerror is None:
                    raise
                error = e.strerror
            self.chan.send_eod(error)

    def op_put(self, req):
        path = self._resolve(req.get("path"), follow=False)
        size = _opt_int(req.get("size"), "size")
        mode = _opt_int(req.get("mode"), "mode")
        mtime = _opt_int(req.get("mtime"), "mtime")
        if os.path.isdir(path):
            raise FsError("is a directory")
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".lwe-sftp-")
        self.chan.send_msg({"ok": True})

        error, received, done = None, 0, False
        try:
            with os.fdopen(fd, "wb") as f:
                while True:
                    chunk, status = self.chan.recv_chunk()
                    if chunk is None:
                        break
                    if error is None:
                        try:
                            f.write(chunk)
                            received += len(chunk)
                        except OSError as e:
                            error = e.strerror
                if not status.get("ok"):
                    error = error or f"upload aborted by client: {status.get('error')}"
                if error is None and size is not None and received != size:
                    error = f"size mismatch: expected {size}, received {received}"
                if error is None:
                    f.flush()
                    os.fsync(f.fileno())
            if error is None:
                # never let a client create setuid/setgid files
                os.chmod(tmp, (int(mode) & 0o1777) if mode is not None else (0o666 & ~keys.UMASK))
                if mtime is not None:
                    os.utime(tmp, (mtime, mtime))
                os.replace(tmp, path)
                done = True
        except OSError as e:
            if e.strerror is None:
                raise
            error = e.strerror
        finally:
            if not done:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        self.chan.send_msg({"ok": error is None, "error": error})

    def op_mkdir(self, req):
        os.mkdir(self._resolve(req.get("path"), follow=False), 0o777)
        self.chan.send_msg({"ok": True})

    def op_rmdir(self, req):
        os.rmdir(self._resolve(req.get("path"), follow=False))
        self.chan.send_msg({"ok": True})

    def op_remove(self, req):
        path = self._resolve(req.get("path"), follow=False)
        if os.path.isdir(path) and not os.path.islink(path):
            raise FsError("is a directory")
        os.unlink(path)
        self.chan.send_msg({"ok": True})

    def op_rename(self, req):
        os.rename(self._resolve(req.get("src"), follow=False), self._resolve(req.get("dst"), follow=False))
        self.chan.send_msg({"ok": True})


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        log.info("%s:%s: connected", *self.client_address[:2])
        Session(self.request, self.client_address, self.server.config, self.server.host_key,
                self.server.rlwe_params).run()


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, host, port, config, rlwe_params=None):
        self.config = config
        self.rlwe_params = rlwe.resolve_params(rlwe_params, config.dir)
        self.host_key = config.load_host_key()
        if ":" in host:
            self.address_family = socket.AF_INET6
        super().__init__((host, port), _Handler)
