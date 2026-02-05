#!/usr/bin/env bash
set -euo pipefail

# Fixed paths and settings (no CLI args by design)
WORKDIR="/fss/projects/synaps-i/mingdu/ptycho-vit"
CONFIG="/fss/projects/synaps-i/mingdu/ptycho-vit/workspace/300m_from_scratch/config.yaml"
PYTHON_BIN="${WORKDIR}/.venv/bin/python"

# Distributed settings
NPROC_PER_NODE=8
MASTER_PORT=29500
# Set to a positive integer to use only the first N nodes (0 = all nodes)
NUM_NODES=0

# Logging
LOG_DIR="/fss/projects/synaps-i/mingdu/ptycho-vit/workspace/logs"
TMUX_SESSION_BASE="ptycho_train"
# Optional: set to a file with one hostname per line to bypass mgmt query
NODELIST_FILE=""
# NODELIST_FILE="${WORKDIR}/workspace/node_list.txt"

if [[ ! -d "${WORKDIR}" ]]; then
  echo "ERROR: WORKDIR not found: ${WORKDIR}"
  exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: CONFIG not found: ${CONFIG}"
  exit 1
fi

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

NNODES="${#HOST_ARR[@]}"
MASTER_ADDR="${HOST_ARR[0]}"

mkdir -p "${LOG_DIR}"

echo
printf "Hosts (%s):\n" "${NNODES}"
printf '  - %s\n' "${HOST_ARR[@]}"

echo
echo "Launching training..."

echo "Checking for existing tmux sessions..."
EXISTING_SESSIONS=()
for host in "${HOST_ARR[@]}"; do
  session_name="${TMUX_SESSION_BASE}_${host}"
  if ssh "${host}" "tmux has-session -t \"${session_name}\"" >/dev/null 2>&1; then
    EXISTING_SESSIONS+=("${host}:${session_name}")
  fi
done

if [[ "${#EXISTING_SESSIONS[@]}" -gt 0 ]]; then
  echo "ERROR: Found existing tmux sessions. Refusing to start new training."
  printf '  - %s\n' "${EXISTING_SESSIONS[@]}"
  exit 3
fi

launch_on_node() {
  local host="$1"
  local rank="$2"
  local log_file="${LOG_DIR}/train_${host}.log"

  local session_name="${TMUX_SESSION_BASE}_${host}"

  if [[ "${NNODES}" -eq 1 ]]; then
    ssh "${host}" "bash -s" <<EOF &
set -euo pipefail
cd "${WORKDIR}"
mkdir -p "${LOG_DIR}"
export WANDB_API_KEY=$(cat /home/mingdu/Documents/api_keys/wandb.txt)
tmux new-session -d -s "${session_name}" "bash -lc 'source \"${WORKDIR}/.venv/bin/activate\"; ${PYTHON_BIN} main.py --config \"${CONFIG}\" > \"${log_file}\" 2>&1'"
tmux has-session -t "${session_name}"
EOF
    return 0
  fi

  ssh "${host}" "bash -s" <<EOF &
set -euo pipefail
cd "${WORKDIR}"
mkdir -p "${LOG_DIR}"
export WANDB_API_KEY=$(cat /home/mingdu/Documents/api_keys/wandb.txt)
# Optional: avoid accidental NCCL over ethernet/docker/loopback
export NCCL_SOCKET_IFNAME=^lo,docker0,eth0,eth1
export NCCL_DEBUG=INFO

# NCCL / NVLS isolation + better errors
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
tmux new-session -d -s "${session_name}" "bash -lc 'source \"${WORKDIR}/.venv/bin/activate\"; torchrun \
  --nnodes ${NNODES} \
  --nproc_per_node ${NPROC_PER_NODE} \
  --node_rank ${rank} \
  --master_addr ${MASTER_ADDR} \
  --master_port ${MASTER_PORT} \
  main.py --config \"${CONFIG}\" > \"${log_file}\" 2>&1'"
tmux has-session -t "${session_name}"
EOF
}

for i in "${!HOST_ARR[@]}"; do
  launch_on_node "${HOST_ARR[$i]}" "${i}"
  echo "  started ${HOST_ARR[$i]} (rank ${i})"
  sleep 0.2
  
done

wait

echo "Done. Logs: ${LOG_DIR}"
