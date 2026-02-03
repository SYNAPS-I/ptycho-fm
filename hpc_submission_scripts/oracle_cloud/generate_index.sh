#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/fss/projects/synaps-i/mingdu/ptycho-vit"
REMOTE_WORKDIR="/fss/projects/synaps-i/mingdu/ptycho_data_conversion"
NODELIST_FILE=""
# Logging
LOG_DIR="${WORKDIR}/workspace/logs"
TMUX_SESSION_BASE="ptycho_index"
# Set to a positive integer to use only the first N nodes (0 = all nodes)
NUM_NODES=0

if [[ -n "${NODELIST_FILE}" ]]; then
  if [[ ! -f "${NODELIST_FILE}" ]]; then
    echo "ERROR: NODELIST_FILE not found: ${NODELIST_FILE}"
    exit 1
  fi
  HOSTS="$(awk 'NF {print $1}' "${NODELIST_FILE}")"
else
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
fi

if [[ -z "${HOSTS}" ]]; then
  echo "ERROR: No hosts found"
  exit 1
fi

mapfile -t HOST_ARR <<< "${HOSTS}"
if [[ "${NUM_NODES}" -gt 0 ]]; then
  if [[ "${NUM_NODES}" -gt "${#HOST_ARR[@]}" ]]; then
    echo "ERROR: NUM_NODES (${NUM_NODES}) exceeds available hosts (${#HOST_ARR[@]})"
    exit 2
  fi
  HOST_ARR=("${HOST_ARR[@]:0:${NUM_NODES}}")
fi

mkdir -p "${LOG_DIR}"

printf "Hosts (%s):\n" "${#HOST_ARR[@]}"
printf '  - %s\n' "${HOST_ARR[@]}"

echo "Running index generation on all nodes..."

FAILURES=0
for host in "${HOST_ARR[@]}"; do
  log_file="${LOG_DIR}/create_index_${host}.log"
  echo "  running on ${host}"
  session_name="${TMUX_SESSION_BASE}_${host}"
  if ! ssh "${host}" "bash -s" <<EOF
set -euo pipefail
tmux new-session -d -s "${session_name}" "bash -lc '
  cd \"${REMOTE_WORKDIR}\"
  source \"${REMOTE_WORKDIR}/.venv/bin/activate\"
  python create_index.py \
    --data_root /mnt/localdisk/scratch/simulated_data_cleanedProbe_2/ \
    --output_path /mnt/localdisk/scratch/simulated_data_cleanedProbe_2/index.csv \
    > \"${log_file}\" 2>&1
'"
EOF
  then
    echo "  ERROR: failed on ${host} (see ${log_file})"
    FAILURES=$((FAILURES + 1))
  fi
  sleep 0.2
  
done

if [[ "${FAILURES}" -gt 0 ]]; then
  echo "Done with failures: ${FAILURES}"
  exit 3
fi

echo "Done."
