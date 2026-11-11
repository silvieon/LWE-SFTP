"""Command-line entry points: lscp, lsftp, lsftpd."""

import argparse
import cmd
import getpass
import logging
import os
import posixpath
import shlex
import signal
import stat
import sys
import time

from . import DEFAULT_PORT, __version__, keys, rlwe
from .client import AuthError, Client, HostKeyError, RemoteError
from .transport import HandshakeError, ProtocolError
from .server import Config, Server, default_config_dir

CONNECT_ERRORS = (OSError, AuthError, HostKeyError, HandshakeError, ProtocolError, keys.KeyFileError, ValueError)


# --- shared helpers ----------------------------------------------------------

def _add_conn_args(p):
    p.add_argument("-P", "--port", type=int, default=DEFAULT_PORT, help=f"server port (default {DEFAULT_PORT})")
    p.add_argument("-i", "--identity", help="Ed25519 private key (default ~/.lwe-sftp/id_ed25519, then ~/.ssh/id_ed25519)")
    p.add_argument("--accept-new-host", action="store_true", help="trust unknown host keys without asking (TOFU)")
    p.add_argument("--strict-host", action="store_true", help="refuse hosts not already in known_hosts")
    p.add_argument("--rlwe", metavar="N,Q,K", help="R-LWE parameters (must match the server's; "
                   "default: $LWE_SFTP_RLWE, ~/.lwe-sftp/rlwe_params.json, then the packaged set)")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")


def _split_userhost(s):
    user, _, host = s.rpartition("@")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    return (user or getpass.getuser()), host


def _confirm_host(label, fp):
    if not sys.stdin.isatty():
        return False
    sys.stderr.write(f"The authenticity of host '{label}' can't be established.\n"
                     f"ED25519 host key fingerprint is {fp}.\n")
    while True:
        sys.stderr.write("Are you sure you want to continue connecting (yes/no)? ")
        sys.stderr.flush()
        ans = sys.stdin.readline().strip().lower()
        if ans in ("yes", "no", ""):
            return ans == "yes"


def _password_prompt(user, host):
    if not sys.stdin.isatty():
        return None
    try:
        return getpass.getpass(f"{user}@{host}'s password: ")
    except (EOFError, KeyboardInterrupt):
        return None


def _make_client(user, host, args):
    policy = "accept-new" if args.accept_new_host else "strict" if args.strict_host else "ask"
    return Client(
        host, args.port, user, identity=args.identity, password_prompt=_password_prompt,
        passphrase_prompt=lambda p: getpass.getpass(f"Enter passphrase for key '{p}': "),
        host_key_policy=policy, confirm_host=_confirm_host, rlwe_params=args.rlwe,
    )


def _human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024


class Progress:
    """scp-style single-line progress on stderr."""

    def __init__(self, enabled):
        self.enabled = enabled and sys.stderr.isatty()

    def for_file(self, name):
        if not self.enabled:
            return None
        start = time.monotonic()
        label = os.path.basename(name.rstrip("/")) or name

        def cb(done, total):
            elapsed = max(time.monotonic() - start, 1e-6)
            pct = 100 if total == 0 else int(done * 100 / total)
            sys.stderr.write(f"\r{label[:40]:<40} {pct:3d}% {_human(done):>9} {_human(done / elapsed):>9}/s")
            if done >= total:
                sys.stderr.write("\n")
            sys.stderr.flush()
        cb(0, 1)
        return cb


def _rjoin(base, path):
    """Resolve a remote path against a remote cwd ('/' is the user's root)."""
    return posixpath.normpath(posixpath.join(base, path)) if path else base


# --- lscp ------------------------------------------------------------------

def _parse_spec(s):
    """scp rule: 'host:path' is remote if a ':' appears before any '/'."""
    head, sep, path = s.partition(":")
    if sep and head and "/" not in head:
        user, host = _split_userhost(head)
        return (user, host, path or ".")
    return None


def lscp_main(argv=None):
    p = argparse.ArgumentParser(
        prog="lscp", description="Copy files over an LWE-SFTP (post-quantum hybrid) connection, scp-style.",
        epilog="examples:  lscp file.txt alice@host:docs/    lscp -r alice@host:project ./backup",
    )
    p.add_argument("-r", "--recursive", action="store_true", help="copy directories recursively")
    p.add_argument("-p", "--preserve", action="store_true", help="preserve modification times and modes")
    p.add_argument("-q", "--quiet", action="store_true", help="no progress meter")
    _add_conn_args(p)
    p.add_argument("source", nargs="+")
    p.add_argument("target")
    args = p.parse_args(argv)

    srcs = [(s, _parse_spec(s)) for s in args.source]
    dst = _parse_spec(args.target)
    remote_srcs = [r for _, r in srcs if r]
    if dst and remote_srcs:
        p.error("remote-to-remote copies are not supported")
    if not dst and not remote_srcs:
        p.error("one side must be remote (user@host:path)")
    if not dst and len(remote_srcs) != len(srcs):
        p.error("cannot mix local and remote sources")
    endpoints = {(u, h) for u, h, _ in remote_srcs} if not dst else {dst[:2]}
    if len(endpoints) != 1:
        p.error("all remote sources must be on the same user@host")
    user, host = endpoints.pop()

    progress = Progress(not args.quiet)
    failures = []

    def fail(path, err):
        failures.append(path)
        print(f"lscp: {path}: {err}", file=sys.stderr)

    try:
        client = _make_client(user, host, args).connect()
    except CONNECT_ERRORS as e:
        print(f"lscp: {host}: {e}", file=sys.stderr)
        return 255

    with client:
        try:
            if dst:
                _upload(client, [s for s, _ in srcs], dst[2], args, progress, fail)
            else:
                _download(client, [r[2] for _, r in srcs], args.target, args, progress, fail)
        except (RemoteError, OSError, ProtocolError) as e:
            fail(args.target, e)
    return 1 if failures else 0


def _upload(client, sources, target, args, progress, fail):
    target = _rjoin("/", target)
    target_is_dir = client.is_dir(target)
    if len(sources) > 1 and not target_is_dir:
        fail(target, "not a directory")
        return
    for src in sources:
        name = os.path.basename(src.rstrip("/")) or src
        dest = posixpath.join(target, name) if target_is_dir else target
        try:
            if os.path.isdir(src):
                if not args.recursive:
                    fail(src, "is a directory (use -r)")
                    continue
                client.put_tree(src, dest, args.preserve, lambda l, r: progress.for_file(l), fail)
            else:
                client.put(src, dest, args.preserve, progress.for_file(src))
        except (RemoteError, OSError) as e:
            fail(src, e)


def _download(client, sources, target, args, progress, fail):
    target_is_dir = os.path.isdir(target)
    if len(sources) > 1 and not target_is_dir:
        fail(target, "not a directory")
        return
    for src in sources:
        rsrc = _rjoin("/", src)
        name = posixpath.basename(rsrc) or "root"
        dest = os.path.join(target, name) if target_is_dir else target
        try:
            st = client.stat(rsrc)
            if st["type"] == "dir":
                if not args.recursive:
                    fail(src, "is a directory (use -r)")
                    continue
                client.get_tree(rsrc, dest, args.preserve, lambda r, l: progress.for_file(r), fail)
            else:
                client.get(rsrc, dest, args.preserve, progress.for_file(rsrc))
        except (RemoteError, OSError) as e:
            fail(src, e)


# --- lsftp -------------------------------------------------------------------

def _fmt_entry(e):
    kind = {"dir": "d", "file": "-"}.get(e["type"], "?")
    perms = stat.filemode(e["mode"])[1:]
    when = time.strftime("%b %d %H:%M", time.localtime(e["mtime"]))
    return f"{kind}{perms} {e['size']:>12} {when} {e['name']}{'/' if e['type'] == 'dir' else ''}"


class Shell(cmd.Cmd):
    intro = None
    prompt = "lsftp> "

    def __init__(self, client, cwd="/"):
        super().__init__()
        self.client = client
        self.cwd = cwd
        self.failed = False
        self.progress = Progress(True)

    def r(self, path):
        return _rjoin(self.cwd, path)

    def onecmd(self, line):
        line = line.strip()
        ignore_err = line.startswith("-")
        if ignore_err:
            line = line[1:]
        try:
            return super().onecmd(line)
        except (RemoteError, OSError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            if not ignore_err:
                self.failed = True
            return False

    def emptyline(self):
        return False

    def default(self, line):
        print(f"unknown command: {line.split()[0]} (try 'help')", file=sys.stderr)
        self.failed = True

    @staticmethod
    def _args(arg):
        parts = shlex.split(arg)
        flags = {a for a in parts if a.startswith("-") and len(a) > 1}
        return [a for a in parts if a not in flags], flags

    def do_ls(self, arg):
        """ls [path] - list a remote directory"""
        paths, _ = self._args(arg)
        path = self.r(paths[0] if paths else "")
        st = self.client.stat(path)
        entries = self.client.listdir(path) if st["type"] == "dir" else [{**st, "name": posixpath.basename(path)}]
        for e in entries:
            print(_fmt_entry(e))

    do_dir = do_ls

    def do_cd(self, arg):
        """cd [path] - change remote directory ('/' is your root)"""
        path = self.r(arg.strip() or "/")
        if self.client.stat(path)["type"] != "dir":
            raise RemoteError(f"{path}: not a directory")
        self.cwd = path

    def do_pwd(self, arg):
        """pwd - print remote directory"""
        print(f"Remote working directory: {self.cwd}")

    def do_lcd(self, arg):
        """lcd path - change local directory"""
        os.chdir(os.path.expanduser(arg.strip() or "~"))

    def do_lpwd(self, arg):
        """lpwd - print local directory"""
        print(f"Local working directory: {os.getcwd()}")

    def do_lls(self, arg):
        """lls [path] - list a local directory"""
        path = arg.strip() or "."
        for name in sorted(os.listdir(path)):
            print(name + ("/" if os.path.isdir(os.path.join(path, name)) else ""))

    def do_get(self, arg):
        """get [-r] [-p] remote [local] - download"""
        paths, flags = self._args(arg)
        if not paths:
            raise ValueError("usage: get [-r] [-p] remote [local]")
        src = self.r(paths[0])
        dest = paths[1] if len(paths) > 1 else posixpath.basename(src) or "root"
        if os.path.isdir(dest):
            dest = os.path.join(dest, posixpath.basename(src))
        preserve = "-p" in flags
        if self.client.stat(src)["type"] == "dir":
            if "-r" not in flags:
                raise ValueError(f"{src} is a directory (use get -r)")
            self.client.get_tree(src, dest, preserve, lambda r, l: self.progress.for_file(r))
        else:
            self.client.get(src, dest, preserve, self.progress.for_file(src))

    def do_put(self, arg):
        """put [-r] [-p] local [remote] - upload"""
        paths, flags = self._args(arg)
        if not paths:
            raise ValueError("usage: put [-r] [-p] local [remote]")
        src = os.path.expanduser(paths[0])
        name = os.path.basename(src.rstrip("/"))
        dest = self.r(paths[1] if len(paths) > 1 else name)
        if len(paths) > 1 and self.client.is_dir(dest):
            dest = posixpath.join(dest, name)
        preserve = "-p" in flags
        if os.path.isdir(src):
            if "-r" not in flags:
                raise ValueError(f"{src} is a directory (use put -r)")
            self.client.put_tree(src, dest, preserve, lambda l, r: self.progress.for_file(l))
        else:
            self.client.put(src, dest, preserve, self.progress.for_file(src))

    def do_mkdir(self, arg):
        """mkdir path - create remote directory"""
        self.client.mkdir(self.r(arg.strip()))

    def do_rmdir(self, arg):
        """rmdir path - remove empty remote directory"""
        self.client.rmdir(self.r(arg.strip()))

    def do_rm(self, arg):
        """rm path... - remove remote files"""
        for p in shlex.split(arg):
            self.client.remove(self.r(p))

    def do_rename(self, arg):
        """rename old new - rename/move a remote file"""
        paths, _ = self._args(arg)
        if len(paths) != 2:
            raise ValueError("usage: rename old new")
        self.client.rename(self.r(paths[0]), self.r(paths[1]))

    do_mv = do_rename

    def do_exit(self, arg):
        """exit - close the session"""
        return True

    do_quit = do_bye = do_exit

    def do_EOF(self, arg):
        print()
        return True


def lsftp_main(argv=None):
    p = argparse.ArgumentParser(prog="lsftp", description="Interactive LWE-SFTP client (sftp-style).")
    p.add_argument("-b", "--batch", metavar="FILE",
                   help="run commands from FILE ('-' = stdin); stop at the first error unless the line starts with '-'")
    _add_conn_args(p)
    p.add_argument("destination", help="[user@]host[:path]")
    args = p.parse_args(argv)

    spec = _parse_spec(args.destination)
    user, host, path = spec if spec else (*_split_userhost(args.destination), "/")

    try:
        client = _make_client(user, host, args).connect()
    except CONNECT_ERRORS as e:
        print(f"lsftp: {host}: {e}", file=sys.stderr)
        return 255

    with client:
        if not args.batch:
            print(f"Connected to {host} as {user} (LWE-SFTP, hybrid R-LWE + ML-KEM-768 + X25519).")
        stdin = None
        if args.batch:
            stdin = sys.stdin if args.batch == "-" else open(args.batch)
        try:
            shell = Shell(client, _rjoin("/", path))
            if args.batch:
                for line in stdin:
                    if shell.onecmd(line) or shell.failed:
                        break
            else:
                shell.cmdloop()
        except KeyboardInterrupt:
            print()
        except ProtocolError as e:
            print(f"lsftp: connection error: {e}", file=sys.stderr)
            return 1
        finally:
            if stdin not in (None, sys.stdin):
                stdin.close()
    return 1 if shell.failed else 0


# --- lsftpd ----------------------------------------------------------------

def lsftpd_main(argv=None):
    p = argparse.ArgumentParser(prog="lsftpd", description="LWE-SFTP server and user administration.")
    p.add_argument("-c", "--config-dir", default=None, help=f"config directory (default {default_config_dir()})")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the server")
    s.add_argument("-l", "--listen", default="0.0.0.0", help="address to bind (default 0.0.0.0)")
    s.add_argument("-p", "--port", type=int, default=DEFAULT_PORT)
    s.add_argument("-v", "--verbose", action="store_true")
    s.add_argument("--rlwe", metavar="N,Q,K", help="R-LWE parameters (default: $LWE_SFTP_RLWE, "
                   "<config-dir>/rlwe_params.json, then the packaged set)")

    a = sub.add_parser("adduser", help="add or update a user")
    a.add_argument("name")
    a.add_argument("--root", help="directory the user is confined to (required for new users)")
    a.add_argument("--password", action="store_true", help="prompt for a password")
    a.add_argument("--authorized-key", action="append", default=[], metavar="KEY",
                   help="Ed25519 public key file or 'ssh-ed25519 AAAA...' line (repeatable)")

    d = sub.add_parser("deluser", help="remove a user")
    d.add_argument("name")

    sub.add_parser("users", help="list users")
    sub.add_parser("fingerprint", help="print the host key fingerprint (generating the key if needed)")
    sub.add_parser("host-key", help="print the host public key line (for known_hosts)")

    svc = sub.add_parser("install-service", help="print a systemd unit for this installation",
                         description="Prints a unit file; install with:  lsftpd install-service | "
                                     "sudo tee /etc/systemd/system/lsftpd.service")
    svc.add_argument("--user", default=getpass.getuser(), help="account the server runs as (default: you)")
    svc.add_argument("-l", "--listen", default="0.0.0.0")
    svc.add_argument("-p", "--port", type=int, default=DEFAULT_PORT)

    args = p.parse_args(argv)
    config = Config(os.path.abspath(os.path.expanduser(args.config_dir or default_config_dir())))

    if args.cmd == "install-service":
        config.load_host_key()  # make sure the key exists and is owned by the service user
        print(SERVICE_TEMPLATE.format(
            user=args.user, python=sys.executable, config=shlex.quote(config.dir),
            listen=args.listen, port=args.port,
        ), end="")
        return 0

    if args.cmd == "host-key":
        print(keys.public_to_openssh(config.load_host_key().public_key()))
        return 0

    if args.cmd == "serve":
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                            format="%(asctime)s %(levelname)s %(message)s")
        try:
            server = Server(args.listen, args.port, config, args.rlwe)
        except ValueError as e:
            print(f"lsftpd: {e}", file=sys.stderr)
            return 1
        except OSError as e:
            print(f"lsftpd: cannot listen on {args.listen}:{args.port}: {e.strerror}", file=sys.stderr)
            return 1

        def _on_term(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, _on_term)
        print(f"lsftpd {__version__} listening on {args.listen}:{args.port}  "
              f"host key {keys.fingerprint(server.host_key.public_key())}  R-LWE {server.rlwe_params.name}", flush=True)
        if not config.load_users():
            print(f"warning: no users yet; add one with: lsftpd -c {config.dir} adduser NAME --root DIR --authorized-key ~/.ssh/id_ed25519.pub",
                  file=sys.stderr)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nshutting down")
        finally:
            server.server_close()
        return 0

    if args.cmd == "fingerprint":
        print(keys.fingerprint(config.load_host_key().public_key()))
        return 0

    users = config.load_users()
    if args.cmd == "users":
        for name, rec in sorted(users.items()):
            methods = (["password"] if rec.get("password") else []) + [f"{len(rec.get('authorized_keys', []))} key(s)"]
            print(f"{name}\troot={rec.get('root')}\t{', '.join(methods)}")
        return 0

    if args.cmd == "deluser":
        if users.pop(args.name, None) is None:
            print(f"no such user: {args.name}", file=sys.stderr)
            return 1
        config.save_users(users)
        return 0

    # adduser
    rec = users.get(args.name, {})
    if args.root:
        rec["root"] = os.path.abspath(os.path.expanduser(args.root))
        os.makedirs(rec["root"], exist_ok=True)
    if not rec.get("root"):
        p.error("--root is required for a new user")
    akeys = rec.setdefault("authorized_keys", [])
    for spec in args.authorized_key:
        # '-' = stdin; a path = a .pub or authorized_keys-style file; otherwise a literal key line
        if spec == "-":
            lines = sys.stdin.read().splitlines()
        elif os.path.exists(os.path.expanduser(spec)):
            with open(os.path.expanduser(spec)) as f:
                lines = f.read().splitlines()
        else:
            lines = [spec]
        found = [(ln.strip(), k) for ln in lines if (k := keys.parse_openssh_public(ln))]
        if not found:
            p.error(f"no Ed25519 public key found in {spec!r}")
        for line, pub in found:
            if all(keys.raw(k) != keys.raw(pub) for k in map(keys.parse_openssh_public, akeys) if k):
                akeys.append(line)
    if args.password:
        pw = getpass.getpass(f"password for {args.name}: ")
        if pw != getpass.getpass("again: "):
            print("passwords do not match", file=sys.stderr)
            return 1
        rec["password"] = keys.hash_password(pw)
    if not rec.get("password") and not rec.get("authorized_keys"):
        p.error("give --password and/or --authorized-key")
    users[args.name] = rec
    config.save_users(users)
    print(f"user {args.name}: root={rec['root']}")
    return 0


def _entry(fn):
    def run():
        try:
            sys.exit(fn())
        except KeyboardInterrupt:
            sys.exit(130)
    return run


SERVICE_TEMPLATE = """\
[Unit]
Description=LWE-SFTP server (post-quantum hybrid file transfer)
After=network-online.target
Wants=network-online.target

[Service]
User={user}
ExecStart={python} -m lwe_sftp.cli lsftpd -c {config} serve -l {listen} -p {port}
Restart=on-failure
NoNewPrivileges=yes
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
"""

lscp = _entry(lscp_main)
lsftp = _entry(lsftp_main)
lsftpd = _entry(lsftpd_main)

if __name__ == "__main__":
    # python -m lwe_sftp.cli {lscp|lsftp|lsftpd} ...  (used by the systemd unit; stable across PATH changes)
    tools = {"lscp": lscp, "lsftp": lsftp, "lsftpd": lsftpd}
    if len(sys.argv) < 2 or sys.argv[1] not in tools:
        sys.exit("usage: python -m lwe_sftp.cli {lscp|lsftp|lsftpd} [args...]")
    name = sys.argv.pop(1)
    sys.argv[0] = name
    tools[name]()
