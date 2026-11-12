import os
import socket
import subprocess
import sys
import threading

import numpy as np
import pytest

from lwe_sftp import keys, rlwe
from lwe_sftp.transport import ProtocolError
from lwe_sftp.client import AuthError, Client, HostKeyError, RemoteError
from lwe_sftp.server import Config, Server

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- R-LWE -------------------------------------------------------------------

def test_rlwe_agreement_many_runs():
    for _ in range(100):
        m, s, p = rlwe.initiate()
        p_r, hint, k_r = rlwe.respond(m, p)
        k_i = rlwe.finish(s, p_r, rlwe.decode_bits(rlwe.encode_bits(hint)))
        assert np.array_equal(k_i, k_r)


def test_rlwe_poly_mul_matches_naive():
    a, b = rlwe.sample_uniform(), rlwe.sample_cbd()
    naive = np.zeros(rlwe.N, dtype=object)
    for i in np.nonzero(b)[0]:
        rolled = np.concatenate([-a[rlwe.N - i:], a[: rlwe.N - i]]).astype(object)
        naive += int(b[i]) * rolled
    assert np.array_equal(rlwe.poly_mul(a, b), (naive % rlwe.Q).astype(np.int64))


def test_rlwe_decode_rejects_bad_input():
    with pytest.raises(ValueError):
        rlwe.decode_poly(b"\x00" * 10)
    with pytest.raises(ValueError):
        rlwe.decode_poly(np.full(rlwe.N, rlwe.Q, dtype="<u4").tobytes())


# --- end to end ----------------------------------------------------------------

@pytest.fixture
def env(tmp_path):
    cfg = Config(str(tmp_path / "srv"))
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "outside.txt").write_text("secret")
    user_key = keys.generate_private_key(str(tmp_path / "id_ed25519"))
    cfg.save_users({
        "alice": {
            "root": str(root),
            "password": keys.hash_password("hunter2"),
            "authorized_keys": [keys.public_to_openssh(user_key.public_key())],
        }
    })
    server = Server("127.0.0.1", 0, cfg)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield {"tmp": tmp_path, "root": root, "port": server.server_address[1], "server": server,
           "key": str(tmp_path / "id_ed25519"), "kh": str(tmp_path / "known_hosts")}
    server.shutdown()
    server.server_close()


def connect(env, **kw):
    opts = dict(user="alice", identity=env["key"], known_hosts=env["kh"], host_key_policy="accept-new")
    opts.update(kw)
    return Client("127.0.0.1", env["port"], **opts).connect()


def test_put_get_roundtrip(env):
    data = os.urandom(300_000)  # several chunks
    src = env["tmp"] / "src.bin"
    src.write_bytes(data)
    with connect(env) as c:
        c.put(str(src), "/up.bin")
        assert (env["root"] / "up.bin").read_bytes() == data
        assert c.stat("up.bin")["size"] == len(data)
        c.get("/up.bin", str(env["tmp"] / "down.bin"))
    assert (env["tmp"] / "down.bin").read_bytes() == data


def test_empty_file_and_dir_ops(env):
    src = env["tmp"] / "empty"
    src.write_bytes(b"")
    with connect(env) as c:
        c.mkdir("/d")
        c.put(str(src), "/d/e")
        assert [e["name"] for e in c.listdir("/d")] == ["e"]
        c.rename("/d/e", "/d/f")
        c.remove("/d/f")
        c.rmdir("/d")
        assert c.try_stat("/d") is None


def test_recursive_tree(env):
    tree = env["tmp"] / "tree"
    (tree / "a" / "b").mkdir(parents=True)
    (tree / "a" / "b" / "x.txt").write_text("x")
    (tree / "y.txt").write_text("y")
    with connect(env) as c:
        c.put_tree(str(tree), "/tree")
        c.get_tree("/tree", str(env["tmp"] / "back"))
    assert (env["tmp"] / "back" / "a" / "b" / "x.txt").read_text() == "x"
    assert (env["tmp"] / "back" / "y.txt").read_text() == "y"


def test_cannot_escape_root(env):
    os.symlink(env["tmp"] / "outside.txt", env["root"] / "link")
    with connect(env) as c:
        # '..' is clamped at the root, so this is just /outside.txt inside the root
        with pytest.raises(RemoteError):
            c.get("/../outside.txt", str(env["tmp"] / "x"))
        with pytest.raises(RemoteError, match="permission denied"):
            c.get("/link", str(env["tmp"] / "x"))
        # removing the symlink removes the link, not its target
        c.remove("/link")
    assert (env["tmp"] / "outside.txt").read_text() == "secret"


def test_password_auth_and_failure(env, monkeypatch):
    monkeypatch.setenv("HOME", str(env["tmp"] / "home"))  # no default identities
    monkeypatch.delenv("LWE_SFTP_DIR", raising=False)
    with connect(env, identity=None, password_prompt=lambda u, h: "hunter2") as c:
        assert c.listdir("/") == []
    with pytest.raises(AuthError):
        connect(env, identity=None, password_prompt=lambda u, h: "wrong")


def test_unknown_user_rejected(env):
    with pytest.raises(AuthError):
        connect(env, user="mallory")


def test_host_key_change_detected(env):
    connect(env).close()  # records the key
    with open(env["kh"]) as f:
        line = f.read()
    other = keys.public_to_openssh(keys.generate_private_key(str(env["tmp"] / "other_host")).public_key())
    with open(env["kh"], "w") as f:
        f.write(line.split(" ", 1)[0] + " " + other + "\n")
    with pytest.raises(HostKeyError, match="CHANGED"):
        connect(env)


def test_strict_host_policy_refuses_unknown(env):
    with pytest.raises(HostKeyError):
        connect(env, host_key_policy="strict")


def test_record_tampering_and_replay_detected():
    from lwe_sftp.transport import SecureChannel
    a, b = socket.socketpair()
    k1, k2 = os.urandom(32), os.urandom(32)
    tx, rx = SecureChannel(a, k1, k2, b""), SecureChannel(b, k2, k1, b"")
    tx.send_msg({"op": "x"})
    assert rx.recv_msg() == {"op": "x"}

    # capture one record's raw bytes, then deliver it modified
    c, d = socket.socketpair()
    spy = SecureChannel(c, k1, k2, b"")
    spy._tx_seq = 1
    spy.send_msg({"op": "y"})
    raw = d.recv(65536)
    a.sendall(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ProtocolError):
        rx.recv()

    # replaying an old (validly encrypted) record is rejected too: wrong sequence number
    rx2 = SecureChannel(b, k2, k1, b"")
    rx2._rx_seq = 5
    a.sendall(raw)
    with pytest.raises(ProtocolError):
        rx2.recv()


def test_lscp_cli(env):
    f = env["tmp"] / "cli.txt"
    f.write_text("via cli")
    base = [sys.executable, "-c", "from lwe_sftp.cli import lscp; lscp()",
            "-P", str(env["port"]), "-i", env["key"], "--accept-new-host", "-q"]
    cenv = {**os.environ, "LWE_SFTP_DIR": str(env["tmp"] / "cli-home"), "PYTHONPATH": REPO}
    r = subprocess.run(base + [str(f), "alice@127.0.0.1:"], env=cenv, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (env["root"] / "cli.txt").read_text() == "via cli"
    out = env["tmp"] / "cli-out.txt"
    r = subprocess.run(base + ["alice@127.0.0.1:cli.txt", str(out)], env=cenv, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.read_text() == "via cli"
    r = subprocess.run(base + ["alice@127.0.0.1:missing", str(out)], env=cenv, capture_output=True, text=True)
    assert r.returncode == 1


def test_lsftp_batch(env):
    (env["tmp"] / "b.txt").write_text("batch")
    batch = env["tmp"] / "cmds"
    batch.write_text(f"mkdir sub\ncd sub\nput {env['tmp'] / 'b.txt'}\nls\n-rm nope\npwd\n")
    cenv = {**os.environ, "LWE_SFTP_DIR": str(env["tmp"] / "cli-home"), "PYTHONPATH": REPO}
    r = subprocess.run([sys.executable, "-c", "from lwe_sftp.cli import lsftp; lsftp()",
                        "-P", str(env["port"]), "-i", env["key"], "--accept-new-host", "-b", str(batch),
                        "alice@127.0.0.1"], env=cenv, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "b.txt" in r.stdout and "/sub" in r.stdout
    assert (env["root"] / "sub" / "b.txt").read_text() == "batch"


def test_lsftpd_admin_commands(tmp_path):
    k1 = keys.public_to_openssh(keys.generate_private_key(str(tmp_path / "k1")).public_key())
    k2 = keys.public_to_openssh(keys.generate_private_key(str(tmp_path / "k2")).public_key())
    base = [sys.executable, "-m", "lwe_sftp.cli", "lsftpd", "-c", str(tmp_path / "cfg")]
    cenv = {**os.environ, "PYTHONPATH": REPO}

    # several keys on stdin, authorized_keys-style, plus junk lines that must be ignored
    r = subprocess.run(base + ["adduser", "bob", "--root", str(tmp_path / "home"), "--authorized-key", "-"],
                       input=f"# comment\n{k1} laptop\n\n{k2} desktop\n", env=cenv, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert len(Config(str(tmp_path / "cfg")).load_users()["bob"]["authorized_keys"]) == 2

    r = subprocess.run(base + ["host-key"], env=cenv, capture_output=True, text=True)
    assert r.stdout.startswith("ssh-ed25519 ")
    assert (tmp_path / "cfg" / "host_ed25519.pub").read_text() == r.stdout

    r = subprocess.run(base + ["install-service", "--user", "bob", "-p", "4500"], env=cenv, capture_output=True, text=True)
    assert "User=bob" in r.stdout and f"{sys.executable} -m lwe_sftp.cli lsftpd -c" in r.stdout and "-p 4500" in r.stdout


@pytest.mark.parametrize("P", [rlwe.DEFAULT, rlwe.Params(2048, 8380417, 2), rlwe.Params(256, 3329, 2),
                               rlwe.Params(2048, 8380417, 16)])
def test_fft_mul_exact(P):
    for _ in range(5):
        a, b = rlwe.sample_uniform(P), rlwe.sample_cbd(P)
        assert np.array_equal(rlwe.poly_mul(a, b, P), rlwe.poly_mul_schoolbook(a, b, P))
    # worst case for float error: every coefficient at its extreme
    a = np.full(P.n, P.q_half + 1)
    b = np.full(P.n, -P.k) % P.q
    assert np.array_equal(rlwe.poly_mul(a, b, P), rlwe.poly_mul_schoolbook(a, b, P))


def test_params_too_large_for_fft_rejected():
    with pytest.raises(ValueError):
        rlwe.Params(4096, 2 ** 31 - 1, 16)


def test_handshake_rejects_rlwe_param_mismatch():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from lwe_sftp.transport import HandshakeError, client_handshake, server_handshake
    a, b = socket.socketpair()
    errs = []

    def srv():
        try:
            server_handshake(b, Ed25519PrivateKey.generate())
        except HandshakeError as e:
            errs.append(e)

    t = threading.Thread(target=srv)
    t.start()
    with pytest.raises(HandshakeError, match="parameter mismatch"):
        client_handshake(a, rlwe.Params(2048, 8380417, 2))
    t.join()
    assert errs


def test_params_resolution_order(tmp_path, monkeypatch):
    import json as _json
    monkeypatch.delenv(rlwe.ENV_VAR, raising=False)
    assert rlwe.resolve_params() == rlwe.DEFAULT == rlwe.Params.load(rlwe.PACKAGED_PARAMS)
    (tmp_path / rlwe.PARAMS_FILENAME).write_text(_json.dumps({"n": 512, "q": 12289, "k": 4}))
    assert rlwe.resolve_params(config_dir=str(tmp_path)) == rlwe.Params(512, 12289, 4)
    monkeypatch.setenv(rlwe.ENV_VAR, "256,3329,2")
    assert rlwe.resolve_params(config_dir=str(tmp_path)) == rlwe.Params(256, 3329, 2)
    assert rlwe.resolve_params("n=2048,q=12289,k=2", str(tmp_path)) == rlwe.Params(2048, 12289, 2)
    with pytest.raises(ValueError):
        rlwe.Params.parse("1024,12288,4")  # even modulus


def test_client_server_custom_params_and_mismatch(env, monkeypatch):
    # the env fixture's server uses the packaged default; a client on other params is refused clearly
    from lwe_sftp.transport import HandshakeError
    with pytest.raises(HandshakeError, match="parameter mismatch"):
        connect(env, rlwe_params="2048,12289,2")
    # a server and client that agree on non-default params work end to end
    cfg = Config(str(env["tmp"] / "srv"))
    srv = Server("127.0.0.1", 0, cfg, rlwe_params="2048,12289,2")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        c = Client("127.0.0.1", srv.server_address[1], user="alice", identity=env["key"],
                   known_hosts=env["kh"], host_key_policy="accept-new", rlwe_params="2048,12289,2").connect()
        with c:
            assert c.listdir("/") == []
    finally:
        srv.shutdown()
        srv.server_close()
