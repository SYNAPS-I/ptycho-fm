#!/usr/bin/env bash
set -euo pipefail

# Fixed settings (no CLI args by design)
TMUX_SESSION_BASE="ptycho_normdict"

echo "Querying compute nodes via mgmt..."
export NODES_JSON="$(mgmt nodes list --fields role=compute --format json)"

HOSTS="$(python3 - <<'PY'
import json, os

raw = os.environ.get("NODES_JSON")
if not raw:
    raise RuntimeError("NODES_JSON is empty (mgmt query failed).")

data = json.loads(raw)
items = data.get("items") if isinstance(data, dict) else data
if not isinstance(items, list):
    raise RuntimeError(f"Unexpected NODES_JSON shape: {type(items).__name__}")

def get_hostname(n: dict):
    return n.get("hostname") or (n.get("fields") or {}).get("hostname")

def get_status(n: dict) -> str:
    return (n.get("status") or (n.get("fields") or {}).get("status") or "").lower()

preferred = []
for n in items:
    hn = get_hostname(n)
    if not hn:
        continue
    st = get_status(n)
    if st and st not in ("active", "ready", "running"):
        continue
    preferred.append(hn)

hosts = preferred or [hn for hn in (get_hostname(n) for n in items) if hn]
print("\n".join(hosts))
PY
<<< "${NODES_JSON}")"

if [[ -z "${HOSTS}" ]]; then
  echo "ERROR: No hosts found"
  exit 1
fi

mapfile -t HOST_ARR <<< "${HOSTS}"

echo
printf "Hosts (%s):\n" "${#HOST_ARR[@]}"
printf '  - %s\n' "${HOST_ARR[@]}"

echo

echo "Killing tmux sessions with prefix: ${TMUX_SESSION_BASE}_*"

for host in "${HOST_ARR[@]}"; do
  session_name="${TMUX_SESSION_BASE}_${host}"
  if ssh "${host}" "tmux has-session -t \"${session_name}\"" >/dev/null 2>&1; then
    ssh "${host}" "tmux kill-session -t \"${session_name}\""
    echo "  killed ${host}:${session_name}"
  else
    echo "  no session on ${host}"
  fi
  
done

echo "Done."
