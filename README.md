# LWE-SFTP

`scp`/`sftp`-style file transfer over a hybrid post-quantum channel. The key exchange
combines **Ring-LWE** (Ding, Xie & Lin, [ePrint 2012/688](docs/ding-xie-lin-2012-688.pdf)),
**ML-KEM-768** and **X25519**.

| Tool     | Replaces   | What it does                                          |
|----------|------------|-------------------------------------------------------|
| `lscp`   | `scp`      | copy files and directories (`-r`, `-p`, multiple sources) |
| `lsftp`  | `sftp`     | interactive shell, or scripted with `-b batchfile`    |
| `lsftpd` | `sshd`'s sftp-server | the server, plus user administration        |

## Install

Needs Python ≥ 3.10 on every machine. Dependencies (`numpy`, `cryptography` ≥ 50 for ML-KEM)
are pulled in automatically.

### On your laptop / any client

Install straight from the repo with [pipx](https://pipx.pypa.io). It gives the tool its own
virtualenv and puts `lscp`, `lsftp` and `lsftpd` on your PATH. Plain `pip install` into the system
Python is blocked on Arch/EndeavourOS, Debian 12+ and Ubuntu 23.04+.

```sh
sudo pacman -S python-pipx      # Debian/Ubuntu: sudo apt install pipx   Fedora: sudo dnf install pipx
pipx ensurepath                 # once; puts ~/.local/bin on PATH (open a new shell afterwards)

pipx install "git+https://github.com/silvieon/LWE-SFTP"
```

* A specific branch, tag or commit: append `@<ref>`, e.g. `".../LWE-SFTP@v0.2"`.
* Passphrase-protected keys: `pipx inject lwe-sftp bcrypt`.
* Upgrade: `pipx upgrade lwe-sftp`. Remove: `pipx uninstall lwe-sftp`.
* From a local checkout instead: `pipx install .`, or `pip install -e '.[test]'` in a venv for development.
* For every account on the machine (pipx ≥ 1.5): `sudo pipx install --global "git+https://github.com/silvieon/LWE-SFTP"`.

You need an Ed25519 key. Your existing `~/.ssh/id_ed25519` is used automatically. If you don't
have one: `ssh-keygen -t ed25519`.

### On your servers: the one-command way

If you can already `ssh` into the servers, run this from a clone of the repo on your laptop:

```sh
tools/deploy.sh server1 alice@server2 server3
```

For each host, over your normal SSH connection, it:

1. `pipx install`s lwe-sftp for your login user (pipx must already be on the server),
2. authorizes your public key with your home directory as the root (`lscp f server:` lands in `~`, like scp),
3. installs and starts `/etc/systemd/system/lsftpd.service` (sudo will prompt),
4. reads the server's host key **over SSH** and pins it in `~/.lwe-sftp/known_hosts`, so the
   first `lscp`/`lsftp` connection is already verified and you never see a trust-on-first-use prompt.

It is safe to re-run, e.g. to upgrade. Options: `-k key.pub`, `-p port`, `-s <pip source>`
(e.g. `-s "git+https://github.com/silvieon/LWE-SFTP@v0.2"`), and `--no-service` to skip
systemd/sudo. Then open the port in each server's firewall:

```sh
sudo firewall-cmd --add-port=4422/tcp --permanent && sudo firewall-cmd --reload   # firewalld
sudo ufw allow 4422/tcp                                                              # ufw
```

Pin under the same name you'll type later. The host key is recorded as `[name]:port`, and
`lscp`/`lsftp` don't read `~/.ssh/config`. If `server1` is only an SSH alias, deploy with a
name that resolves in DNS or `/etc/hosts`.

### On your servers: by hand

```sh
pipx install "git+https://github.com/silvieon/LWE-SFTP"
lsftpd adduser "$USER" --root ~ --authorized-key - < laptop_id_ed25519.pub   # or a file path, or --password
lsftpd install-service | sudo tee /etc/systemd/system/lsftpd.service          # prints a unit for this install
sudo systemctl daemon-reload && sudo systemctl enable --now lsftpd
lsftpd fingerprint        # compare with what lscp shows on its first connection
```

`install-service` takes `--user`, `--listen` and `--port`. It runs the server as you by default.
Logs: `journalctl -u lsftpd -f`. Foreground for testing: `lsftpd serve -v`.

Server state lives in `~/.config/lwe-sftpd/` (or `/etc/lwe-sftpd/` as root, `$LWE_SFTPD_DIR`,
or `-c DIR`):
* `host_ed25519` / `host_ed25519.pub`: created on first run.
* `users.json`: re-read on every login, so `adduser`/`deluser` take effect without a restart.

Every user's files are owned by the account `lsftpd` runs as. Unlike sshd, it doesn't switch to
each user's own system account, which is why the default setup runs it as you, serving your own home.

More users: `lsftpd adduser bob --root /srv/files/bob --password` (needs a server account that
can write there). List them with `lsftpd users`; remove with `lsftpd deluser bob`.

## Usage

```sh
lscp report.pdf server1:docs/
lscp -r server2:project ./backup
lscp -p a.txt b.txt alice@server2:inbox/
lsftp server3
lsftp -b commands.txt server3:/docs
```

| scp / sftp habit       | here                                                  |
|------------------------|-------------------------------------------------------|
| `scp -P port`          | `lscp -P port` (default 4422)                         |
| `scp -i key`           | `lscp -i key` (Ed25519 only)                          |
| `-o StrictHostKeyChecking=accept-new` | `--accept-new-host`                    |
| `-o StrictHostKeyChecking=yes` | `--strict-host`                               |
| `~/.ssh/known_hosts`   | `~/.lwe-sftp/known_hosts` (same line format)          |
| remote paths           | relative to the user's root; `/` *is* the root        |

`lsftp` commands: `ls cd pwd lcd lpwd lls get [-r] [-p] put [-r] [-p] mkdir rmdir rm rename/mv exit`.
In batch mode the first failing command aborts the run; prefix a line with `-` to ignore its failure (same as sftp).

## Protocol

One TCP connection per session. The handshake takes one round trip:

```
C -> S  {x25519 pub, ML-KEM-768 encaps key, R-LWE m, R-LWE P_c}
S -> C  {x25519 pub, ML-KEM ciphertext, R-LWE P_s, R-LWE hint w, Ed25519 host key}
S -> C  Ed25519 signature over SHA-256(transcript)
keys  = HKDF-SHA256(x25519_ss || mlkem_ss || rlwe_bits, salt = transcript hash)
```

After that, every message is a ChaCha20-Poly1305 record with its own key per direction and a
sequence-number nonce. Tampering, replay, reordering and key mismatch all kill the session.
The client checks the server key against `known_hosts` (trust on first use) before it
authenticates with an Ed25519 signature over the session hash, or with a scrypt-checked password.

Because the three shared secrets are combined, the session stays confidential if **any one**
of the three exchanges holds. That means the R-LWE parameters can be tuned or experimented
with (see below) without making the channel weaker than ML-KEM + X25519 alone.

| File | Contents |
|---|---|
| `lwe_sftp/rlwe.py` | ring arithmetic (FFT multiplication), sampling (os.urandom), KeyGen / Sig (HelpRec) / Mod2 (Rec), parameter loading |
| `lwe_sftp/rlwe_params.json` | the R-LWE parameter set in use, generated by the benchmark |
| `tools/bench.py` | parameter sweep, estimator validation, handshake benchmark |
| `tools/deploy.sh` | install + start the server on hosts over SSH |
| `lwe_sftp/transport.py` | hybrid handshake + host-key signature, framing and AEAD record layer |
| `lwe_sftp/keys.py` | key files, known_hosts, scrypt passwords |
| `lwe_sftp/server.py`, `client.py`, `cli.py` | server, client library, commands |

Tests: `pip install -e '.[test]' && pytest`.

## R-LWE parameters

The R-LWE parameters (ring degree `n`, modulus `q`, noise width `k`) are data, not code. They live
in [`lwe_sftp/rlwe_params.json`](lwe_sftp/rlwe_params.json), which is generated by the benchmark
and ships with the package. The file records the constraints and estimates behind the choice.
The current set is **n=1024, q=7681, k=4**:

* estimated security (core-SVP, primal attack): ~252 bits classical / ~228 quantum
* key-mismatch rate: ~2⁻⁸³
* client hello ≈ 7.2 KB, inside TCP's initial window, so the handshake costs one network round trip

### Re-tuning

```sh
python tools/bench.py validate          # sanity-check the estimators (NewHope numbers, measured failures)
python tools/bench.py sweep             # sweep n, q, k: speed, security, failure, size
python tools/bench.py sweep --write     # ...and write the selected set to lwe_sftp/rlwe_params.json
python tools/bench.py handshake 1024,7681,4 2048,8380417,2   # time the whole handshake for chosen sets
```

How `--write` selects a set: drop everything below `--min-classical` (128), above
`--max-log2-fail` (2⁻⁶⁴), below `--min-k` (2), or with a hello bigger than `--max-hello-bytes`
(TCP initial window). Among the rest, sets within `--tolerance` (15%) of the fastest count as
ties, and the most secure tie wins. Change the thresholds to change the outcome. After writing,
run the tests, commit the file, and redeploy: **both ends must use the same set**. A
mismatched peer is refused with a message naming the server's set.

### Overriding without editing the package

Highest priority first:

1. `--rlwe N,Q,K` on `lscp`, `lsftp` or `lsftpd serve`
2. the `LWE_SFTP_RLWE=N,Q,K` environment variable
3. `rlwe_params.json` in `~/.lwe-sftp/` (client) or the server's config directory (same format as the packaged file)
4. the packaged `lwe_sftp/rlwe_params.json`

`lsftpd serve` prints the set it uses on startup.

### What the benchmark found

* **Multiplication method beats constants.** `poly_mul` was an O(n²) `np.convolve` and took
  ~90% of the handshake. It's now an exact float64 FFT negacyclic product, O(n log n): 24×
  faster at n=2048 (4.0 → 0.17 ms). Exactness is checked against the schoolbook version,
  including worst-case inputs, and `Params` rejects sizes where float64 could round wrong.
  Because it isn't an NTT, `q` doesn't need to be NTT-friendly.
* **Coefficient width matters more than n.** With q < 2¹⁶, coefficients go on the wire as 2
  bytes instead of 4. The old default (n=2048, q=8380417, k=2) produced a 23.6 KB hello, past TCP's
  ~14 KB initial window, so every connection paid an extra round trip. On CPU the difference is
  small: with FFT multiplication both sets handshake in ~2–3 ms (2.6 ms old vs 2.15 ms new).
* **Estimator caveats.** The security numbers are the standard "core-SVP" primal-attack estimate
  (they reproduce NewHope's published 282-bit figure within 2 bits). That estimate is good for
  *ranking* parameter sets, not a proof. It ignores the dual attack and attacks on very small
  secrets, which is why k=1 is excluded by default. The failure model is a Gaussian approximation;
  it matches measured mismatch rates within ~20% and errs pessimistic.

## Security notes, read before relying on this

* **This protocol has not been audited.** It's built from standard primitives with the
  usual constructions, but it is a hobby protocol.
* **Plain OpenSSH may already do what you want.** OpenSSH ≥ 9.0 uses `sntrup761x25519` key
  exchange by default, and ≥ 9.9 uses `mlkem768x25519-sha256`. Check yours with `ssh -Q kex`.
  `scp`/`sftp` over a modern OpenSSH is already hybrid post-quantum against record-now-decrypt-later.
* Host and user authentication use Ed25519, which is not post-quantum. That only matters
  against an attacker with a quantum computer *at connection time*; SSH makes the same trade-off.
* The root-directory confinement resolves symlinks before every operation. A local user who can
  create symlinks inside a served directory at the same moment could still race it.
* No rekeying, compression, resume, or remote-to-remote copy.

## History

v0.1 was a classroom FTP-style toy (separate PASV data channel, XOR "encryption" with the
0/1 key bits, which flipped only the lowest bit of each byte). v0.2 replaces the transport and
fixes `Rec` to use the centered representative, which removes a ~4% key-mismatch rate.
