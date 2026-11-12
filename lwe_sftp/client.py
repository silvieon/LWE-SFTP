"""Client library: connect, verify the host key, authenticate, and run file operations."""

import base64
import os
import posixpath
import socket
import stat
import tempfile

from . import DEFAULT_PORT, keys, rlwe
from .transport import CHUNK, client_handshake


class RemoteError(Exception):
    pass


class AuthError(Exception):
    pass


class HostKeyError(Exception):
    pass


def client_dir():
    return os.environ.get("LWE_SFTP_DIR") or os.path.expanduser("~/.lwe-sftp")


def default_identities():
    return [os.path.join(client_dir(), "id_ed25519"), os.path.expanduser("~/.ssh/id_ed25519")]


class Client:
    """
    host_key_policy:
        "ask"         unknown host -> confirm_host(label, fingerprint) decides (TOFU)
        "accept-new"  unknown host -> remember silently
        "strict"      unknown host -> refuse
    A *changed* host key is always refused.
    """

    def __init__(self, host, port=DEFAULT_PORT, user=None, identity=None, password_prompt=None,
                 passphrase_prompt=None, known_hosts=None, host_key_policy="ask", confirm_host=None,
                 timeout=30, rlwe_params=None):
        self.host, self.port, self.user = host, port, user
        self.identity = identity
        self.password_prompt = password_prompt
        self.passphrase_prompt = passphrase_prompt
        self.known_hosts = keys.KnownHosts(known_hosts or os.path.join(client_dir(), "known_hosts"))
        self.host_key_policy = host_key_policy
        self.confirm_host = confirm_host
        self.timeout = timeout
        self.rlwe_params = rlwe.resolve_params(rlwe_params, client_dir())
        self.sock = None
        self.chan = None

    # --- connection ------------------------------------------------------------

    def connect(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        try:
            self.chan, host_key = client_handshake(self.sock, self.rlwe_params)
            self._check_host_key(host_key)
            self._authenticate()
        except BaseException:
            self.close()
            raise
        self.sock.settimeout(None)
        return self

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def __enter__(self):
        return self if self.sock is not None else self.connect()

    def __exit__(self, *exc):
        self.close()

    def _check_host_key(self, host_key):
        label = keys.host_label(self.host, self.port)
        known = self.known_hosts.lookup(label)
        fp = keys.fingerprint(host_key)
        if known:
            if any(keys.raw(k) == keys.raw(host_key) for k in known):
                return
            raise HostKeyError(
                f"HOST KEY FOR {label} HAS CHANGED (now {fp}). Someone may be intercepting the connection.\n"
                f"If the server's key was legitimately regenerated, remove its line from {self.known_hosts.path}."
            )
        if self.host_key_policy == "accept-new" or (
            self.host_key_policy == "ask" and self.confirm_host is not None and self.confirm_host(label, fp)
        ):
            self.known_hosts.add(label, host_key)
            return
        raise HostKeyError(f"host key for {label} is not known ({fp}); refusing to connect")

    def _authenticate(self):
        if self.user is None:
            raise AuthError("no user name given")
        candidates = [self.identity] if self.identity else [p for p in default_identities() if os.path.exists(p)]
        for path in candidates:
            key = keys.load_private_key(path, self.passphrase_prompt)
            sig = key.sign(keys.auth_payload(self.chan.session_id, self.user))
            resp = self._call_raw(op="auth", user=self.user, method="publickey",
                                  key=keys.public_to_openssh(key.public_key()), sig=base64.b64encode(sig).decode())
            if resp.get("ok"):
                return
            if resp.get("error") != "authentication failed":
                raise AuthError(resp.get("error"))
        if self.password_prompt is not None:
            for _ in range(3):
                pw = self.password_prompt(self.user, self.host)
                if pw is None:
                    break
                resp = self._call_raw(op="auth", user=self.user, method="password", password=pw)
                if resp.get("ok"):
                    return
                if resp.get("error") != "authentication failed":
                    raise AuthError(resp.get("error"))
        raise AuthError(f"authentication failed for {self.user}@{self.host}")

    # --- primitives ----------------------------------------------------------

    def _call_raw(self, **req):
        self.chan.send_msg(req)
        return self.chan.recv_msg()

    def _call(self, **req):
        resp = self._call_raw(**req)
        if not resp.get("ok"):
            raise RemoteError(resp.get("error") or "remote error")
        return resp

    def stat(self, path):
        return self._call(op="stat", path=path)

    def try_stat(self, path):
        try:
            return self.stat(path)
        except RemoteError:
            return None

    def is_dir(self, path):
        st = self.try_stat(path)
        return st is not None and st["type"] == "dir"

    def listdir(self, path):
        self.chan.send_msg({"op": "list", "path": path})
        entries = []
        while True:
            resp = self.chan.recv_msg()
            if not resp.get("ok"):
                raise RemoteError(resp.get("error") or "remote error")
            entries.extend(resp.get("entries", []))
            if not resp.get("more"):
                return sorted(entries, key=lambda e: e["name"])

    def mkdir(self, path):
        self._call(op="mkdir", path=path)

    def rmdir(self, path):
        self._call(op="rmdir", path=path)

    def remove(self, path):
        self._call(op="remove", path=path)

    def rename(self, src, dst):
        self._call(op="rename", src=src, dst=dst)

    # --- transfers -----------------------------------------------------------

    def get(self, remote, local, preserve=False, progress=None):
        """Download one file. Written to a temp file and renamed into place only when complete."""
        info = self._call(op="get", path=remote)
        total = info["size"]
        local_dir = os.path.dirname(os.path.abspath(local))
        fd, tmp = tempfile.mkstemp(dir=local_dir, prefix=".lwe-sftp-") if os.path.isdir(local_dir) else (None, None)
        local_error = None if fd is not None else f"{local_dir}: no such directory"
        done, received = False, 0
        try:
            f = os.fdopen(fd, "wb") if fd is not None else None
            try:
                while True:
                    chunk, status = self.chan.recv_chunk()
                    if chunk is None:
                        break
                    if local_error is None:
                        try:
                            f.write(chunk)
                        except OSError as e:
                            local_error = f"{local}: {e.strerror}"
                    received += len(chunk)
                    if progress:
                        progress(received, total)
            finally:
                if f is not None:
                    f.close()
            if not status.get("ok"):
                raise RemoteError(status.get("error") or "download failed")
            if local_error:
                raise OSError(local_error)
            os.chmod(tmp, (info["mode"] & 0o1777) if preserve else (0o666 & ~keys.UMASK))
            if preserve:
                os.utime(tmp, (info["mtime"], info["mtime"]))
            os.replace(tmp, local)
            done = True
        finally:
            if tmp and not done:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    def put(self, local, remote, preserve=False, progress=None):
        with open(local, "rb") as f:
            st = os.fstat(f.fileno())
            if not stat.S_ISREG(st.st_mode):
                raise OSError(f"{local}: not a regular file")
            req = {"op": "put", "path": remote, "size": st.st_size}
            if preserve:
                req.update(mode=stat.S_IMODE(st.st_mode), mtime=st.st_mtime)
            self._call(**req)
            sent, error = 0, None
            try:
                while chunk := f.read(CHUNK):
                    self.chan.send_data(chunk)
                    sent += len(chunk)
                    if progress:
                        progress(sent, st.st_size)
            except OSError as e:
                if e.strerror is None:
                    raise
                error = f"{local}: {e.strerror}"
            self.chan.send_eod(error)
            resp = self.chan.recv_msg()
            if error:
                raise OSError(error)
            if not resp.get("ok"):
                raise RemoteError(resp.get("error") or "upload failed")

    def put_tree(self, local_dir, remote_dir, preserve=False, on_file=None, on_error=None):
        """Recursive upload. remote_dir is created if needed. on_file(local, remote) returns a progress cb."""
        if not self.is_dir(remote_dir):
            self.mkdir(remote_dir)
        for name in sorted(os.listdir(local_dir)):
            lp, rp = os.path.join(local_dir, name), posixpath.join(remote_dir, name)
            try:
                if os.path.isdir(lp):
                    self.put_tree(lp, rp, preserve, on_file, on_error)
                elif os.path.isfile(lp):
                    self.put(lp, rp, preserve, on_file(lp, rp) if on_file else None)
            except (OSError, RemoteError) as e:
                if on_error is None:
                    raise
                on_error(lp, e)

    def get_tree(self, remote_dir, local_dir, preserve=False, on_file=None, on_error=None):
        os.makedirs(local_dir, exist_ok=True)
        for entry in self.listdir(remote_dir):
            rp, lp = posixpath.join(remote_dir, entry["name"]), os.path.join(local_dir, entry["name"])
            try:
                if entry["type"] == "dir":
                    self.get_tree(rp, lp, preserve, on_file, on_error)
                elif entry["type"] == "file":
                    self.get(rp, lp, preserve, on_file(rp, lp) if on_file else None)
            except (OSError, RemoteError) as e:
                if on_error is None:
                    raise
                on_error(rp, e)
