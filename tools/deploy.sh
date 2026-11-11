#!/usr/bin/env bash
# Install lsftpd on servers over your existing SSH access, run it as a systemd service,
# and pin each server's host key in ~/.lwe-sftp/known_hosts (so first connects need no TOFU prompt).
#
#   tools/deploy.sh [options] [user@]host ...
#
# Each server gets: the lwe-sftp package via pipx (for the SSH login user), your public key
# authorized for that user with root = their home dir, and /etc/systemd/system/lsftpd.service.
set -euo pipefail

SOURCE="${LWE_SFTP_SOURCE:-git+https://github.com/silvieon/LWE-SFTP@master}"
PORT=4422
PUBKEY=""
SERVICE=1

usage() {
    cat <<EOF
usage: $0 [options] [user@]host ...

  -k FILE       public key to authorize (default: ~/.lwe-sftp/id_ed25519.pub, then ~/.ssh/id_ed25519.pub)
  -p PORT       port lsftpd listens on (default 4422)
  -s SOURCE     pip install source (default: \$LWE_SFTP_SOURCE or $SOURCE)
  --no-service  install and register the key only; don't touch systemd (no sudo needed)

Hosts are passed straight to ssh, so ~/.ssh/config aliases work for deploying. lscp/lsftp
don't read ~/.ssh/config, though: use a name that resolves (DNS or /etc/hosts) if you want
the pinned host key to match what you type later.
EOF
}

hosts=()
while (($#)); do
    case "$1" in
        -k) PUBKEY="$2"; shift 2 ;;
        -p) PORT="$2"; shift 2 ;;
        -s) SOURCE="$2"; shift 2 ;;
        --no-service) SERVICE=0; shift ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
        *) hosts+=("$1"); shift ;;
    esac
done
((${#hosts[@]})) || { usage >&2; exit 2; }

if [[ -z "$PUBKEY" ]]; then
    for f in ~/.lwe-sftp/id_ed25519.pub ~/.ssh/id_ed25519.pub; do
        [[ -f "$f" ]] && { PUBKEY="$f"; break; }
    done
fi
[[ -n "$PUBKEY" && -f "$PUBKEY" ]] || { echo "no Ed25519 public key found; create one with: ssh-keygen -t ed25519" >&2; exit 1; }

KH="${LWE_SFTP_DIR:-$HOME/.lwe-sftp}/known_hosts"
mkdir -p "$(dirname "$KH")"
touch "$KH"

# remote prelude: pipx's bin dir isn't on PATH for non-interactive ssh sessions
PRE='export PATH="$HOME/.local/bin:$PATH";'
q() { printf '%q' "$1"; }

failed=()
for target in "${hosts[@]}"; do
    host="${target#*@}"
    echo "==> $target"

    if ! ssh "$target" "$PRE"'
        command -v pipx >/dev/null || { echo "pipx is not installed on this host (apt install pipx | pacman -S pipx | dnf install pipx)" >&2; exit 1; }
        pipx install --force --quiet '"$(q "$SOURCE")"' >/dev/null
        lsftpd adduser "$USER" --root "$HOME" --authorized-key -' < "$PUBKEY"; then
        echo "    install failed on $target" >&2; failed+=("$target"); continue
    fi

    if ((SERVICE)); then
        echo "    installing systemd service (sudo may prompt)"
        if ! ssh -t "$target" "$PRE"' lsftpd install-service -p '"$(q "$PORT")"' | sudo tee /etc/systemd/system/lsftpd.service >/dev/null &&
              sudo systemctl daemon-reload && sudo systemctl enable --quiet lsftpd && sudo systemctl restart lsftpd'; then
            echo "    service setup failed on $target" >&2; failed+=("$target"); continue
        fi
    fi

    hostkey="$(ssh "$target" "$PRE lsftpd host-key")"
    label="[$host]:$PORT"
    existing="$(awk -v l="$label" '$1 == l { print $2 " " $3 }' "$KH")"
    if [[ -z "$existing" ]]; then
        echo "$label $hostkey" >> "$KH"
        echo "    pinned host key for $label"
    elif grep -qxF "$hostkey" <<<"$existing"; then
        echo "    host key for $label already pinned"
    else
        echo "    WARNING: $KH has a different key for $label." >&2
        echo "    If this server was reinstalled, delete that line and re-run; otherwise investigate." >&2
        failed+=("$target")
    fi
done

echo
if ((${#failed[@]})); then
    echo "failed: ${failed[*]}" >&2
    exit 1
fi
echo "done. Make sure TCP port $PORT is open in each server's firewall, then e.g.:"
echo "    lsftp $( ((PORT != 4422)) && echo "-P $PORT ")${hosts[0]}"
