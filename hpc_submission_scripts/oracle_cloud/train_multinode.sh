#!/usr/bin/env bash
set -Eeuo pipefail

# Launch a fixed 3-node, 8-GPU-per-node torchrun job from node rank 0.
# The two positional arguments are the SSH hostnames of node ranks 1 and 2.
# All nodes must expose the repository, config, and virtual environment at the
# same absolute paths, and rank 0 must be reachable from the other nodes at
# MASTER_ADDR:MASTER_PORT.

readonly NNODES=3
readonly NPROC_PER_NODE=8

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/config.yaml}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
TRAIN_ENTRYPOINT="${PROJECT_DIR}/hpc_submission_scripts/oracle_cloud/train_rank.py"
MASTER_ADDR="${MASTER_ADDR:-$(hostname -f)}"
MASTER_PORT="${MASTER_PORT:-29500}"
NCCL_DEBUG_VALUE="${NCCL_DEBUG:-WARN}"
NCCL_SOCKET_IFNAME_VALUE="${NCCL_SOCKET_IFNAME:-}"
SKIP_MLFLOW_PREFLIGHT="${SKIP_MLFLOW_PREFLIGHT:-0}"
RUN_STAMP="$(date +%Y%m%d-%H%M%S)"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") NODE_RANK_1 NODE_RANK_2

Run this script on node rank 0. It launches 8 workers locally and 8 workers
on each remote node over SSH, for 24 global ranks.

Optional environment variables:
  PROJECT_DIR             Shared repository path (default: ${PROJECT_DIR})
  CONFIG_PATH             Shared training config (default: ${CONFIG_PATH})
  PYTHON_BIN              Shared Python executable (default: ${PYTHON_BIN})
  MASTER_ADDR             Rank-0 address visible to the other nodes
  MASTER_PORT             Rendezvous port (default: ${MASTER_PORT})
  LOG_DIR                 Per-node log directory
  NCCL_SOCKET_IFNAME      Interface(s) NCCL should use
  NCCL_DEBUG              NCCL log level (default: ${NCCL_DEBUG_VALUE})
  SKIP_MLFLOW_PREFLIGHT   Set to 1 to skip the authenticated MLflow query

Example:
  MASTER_ADDR=10.0.0.10 \\
  $(basename "$0") compute-2 compute-3

Export the MLflow credentials in the calling shell before launching. Do not
place the password or token directly on this command line.
EOF
}

if [[ $# -ne 2 ]]; then
  usage >&2
  exit 2
fi

readonly LOCAL_HOST="$(hostname -f)"
readonly REMOTE_HOST_1="$1"
readonly REMOTE_HOST_2="$2"
readonly -a SSH=(
  ssh
  -o BatchMode=yes
  -o ConnectTimeout=10
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=4
)

if [[ ! "${MASTER_PORT}" =~ ^[0-9]+$ ]] \
  || (( MASTER_PORT < 1 || MASTER_PORT > 65535 )); then
  echo "ERROR: MASTER_PORT must be an integer from 1 through 65535." >&2
  exit 2
fi

if [[ "${SKIP_MLFLOW_PREFLIGHT}" != "0" \
  && "${SKIP_MLFLOW_PREFLIGHT}" != "1" ]]; then
  echo "ERROR: SKIP_MLFLOW_PREFLIGHT must be 0 or 1." >&2
  exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: Training config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -f "${TRAIN_ENTRYPOINT}" ]]; then
  echo "ERROR: Oracle training entry point not found: ${TRAIN_ENTRYPOINT}" >&2
  exit 1
fi

if ! LAUNCH_IDENTITY="$(
  cd "${PROJECT_DIR}"
  "${PYTHON_BIN}" - "${CONFIG_PATH}" "${RUN_STAMP}" <<'PY'
import sys

from ptycho_fm.utils.config import load_config, resolve_run_name, run_directory

config = load_config(sys.argv[1])
resume = bool(config.get("training", {}).get("resume_from_checkpoint", False))
trainer = config.get("trainer", {}) or {}
has_name = (
    trainer.get("run_name") is not None
    or trainer.get("run_num") is not None
)
if resume or has_name:
    run_name = resolve_run_name(config, generate=False)
else:
    config.setdefault("trainer", {})["run_name"] = sys.argv[2]
    run_name = resolve_run_name(config, generate=False)
print("1" if resume else "0")
print(run_name)
print(run_directory(config).absolute())
PY
)"; then
  echo "ERROR: Could not resolve the model run name from ${CONFIG_PATH}." >&2
  exit 1
fi
mapfile -t LAUNCH_FIELDS <<<"${LAUNCH_IDENTITY}"
RESUME_MODE="${LAUNCH_FIELDS[0]:-}"
MODEL_RUN_NAME="${LAUNCH_FIELDS[1]:-}"
MODEL_RUN_DIR="${LAUNCH_FIELDS[2]:-}"
readonly RESUME_MODE MODEL_RUN_NAME MODEL_RUN_DIR
if [[ -z "${RESUME_MODE}" || -z "${MODEL_RUN_NAME}" || -z "${MODEL_RUN_DIR}" ]]; then
  echo "ERROR: Launcher identity is incomplete." >&2
  exit 1
fi

if [[ -z "${LOG_DIR:-}" ]]; then
  if [[ "${RESUME_MODE}" == "1" ]]; then
    LOG_DIR="${PROJECT_DIR}/workspace/oracle_cloud_logs/${MODEL_RUN_NAME}/${RUN_STAMP}"
  else
    LOG_DIR="${PROJECT_DIR}/workspace/oracle_cloud_logs/${MODEL_RUN_NAME}"
  fi
fi
readonly LOG_DIR

mkdir -p "${LOG_DIR}"

check_local_node() {
  local gpu_count
  gpu_count="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
  if (( gpu_count < NPROC_PER_NODE )); then
    echo "ERROR: ${LOCAL_HOST} exposes ${gpu_count} GPUs; ${NPROC_PER_NODE} are required." >&2
    exit 1
  fi

  "${PYTHON_BIN}" - "${MASTER_ADDR}" "${MASTER_PORT}" <<'PY'
import socket
import sys

address = sys.argv[1]
port = int(sys.argv[2])
socket.getaddrinfo(address, port)
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((address, port))
PY
}

check_remote_node() {
  local host="$1"
  local gpu_count

  gpu_count="$(
    "${SSH[@]}" "${host}" bash -s -- \
      "${PROJECT_DIR}" "${CONFIG_PATH}" "${PYTHON_BIN}" "${TRAIN_ENTRYPOINT}" \
      "${MASTER_ADDR}" "${NPROC_PER_NODE}" <<'REMOTE'
set -Eeuo pipefail
project_dir="$1"
config_path="$2"
python_bin="$3"
train_entrypoint="$4"
master_addr="$5"
required_gpus="$6"

[[ -d "${project_dir}" ]] || {
  echo "missing project directory: ${project_dir}" >&2
  exit 10
}
[[ -f "${config_path}" ]] || {
  echo "missing config: ${config_path}" >&2
  exit 11
}
[[ -x "${python_bin}" ]] || {
  echo "missing Python executable: ${python_bin}" >&2
  exit 12
}
[[ -f "${train_entrypoint}" ]] || {
  echo "missing Oracle training entry point: ${train_entrypoint}" >&2
  exit 14
}

"${python_bin}" -c \
  'import socket, sys; socket.getaddrinfo(sys.argv[1], None)' \
  "${master_addr}"

gpu_count="$("${python_bin}" -c 'import torch; print(torch.cuda.device_count())')"
if (( gpu_count < required_gpus )); then
  echo "only ${gpu_count} GPUs visible; ${required_gpus} are required" >&2
  exit 13
fi
printf '%s\n' "${gpu_count}"
REMOTE
  )" || {
    echo "ERROR: Preflight failed on ${host}." >&2
    echo "Authenticate with SSH once manually and verify the shared paths." >&2
    exit 1
  }

  echo "  ${host}: ${gpu_count} GPUs, paths and MASTER_ADDR verified"
}

stage_resume_checkpoint() {
  if [[ "${RESUME_MODE}" != "1" ]]; then
    return
  fi

  local checkpoint_model="${MODEL_RUN_DIR}/checkpoint_model.pth"
  local checkpoint_state="${MODEL_RUN_DIR}/checkpoint.state"
  local host
  local source_file

  if ! command -v rsync >/dev/null 2>&1; then
    echo "ERROR: rsync is required to stage resume checkpoints." >&2
    exit 1
  fi

  for source_file in "${checkpoint_model}" "${checkpoint_state}"; do
    if [[ ! -s "${source_file}" ]]; then
      echo "ERROR: Resume checkpoint is missing or empty on ${LOCAL_HOST}: ${source_file}" >&2
      exit 1
    fi
  done

  echo "Staging resume checkpoint from ${LOCAL_HOST}:"
  for host in "${REMOTE_HOST_1}" "${REMOTE_HOST_2}"; do
    "${SSH[@]}" "${host}" bash -s -- "${MODEL_RUN_DIR}" <<'REMOTE'
set -Eeuo pipefail
model_run_dir="$1"
mkdir -p "${model_run_dir}"
command -v rsync >/dev/null 2>&1 || {
  echo "rsync is required to stage resume checkpoints" >&2
  exit 22
}
REMOTE

    rsync -ah --info=progress2 \
      -e 'ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=4' \
      "${checkpoint_model}" "${checkpoint_state}" \
      "${host}:${MODEL_RUN_DIR}/"

    "${SSH[@]}" "${host}" bash -s -- \
      "${checkpoint_model}" "${checkpoint_state}" <<'REMOTE'
set -Eeuo pipefail
checkpoint_model="$1"
checkpoint_state="$2"
[[ -s "${checkpoint_model}" ]] || {
  echo "staged model checkpoint is missing or empty: ${checkpoint_model}" >&2
  exit 20
}
[[ -s "${checkpoint_state}" ]] || {
  echo "staged runtime checkpoint is missing or empty: ${checkpoint_state}" >&2
  exit 21
}
REMOTE
    echo "  ${host}: checkpoint_model.pth and checkpoint.state staged"
  done
}

preflight_mlflow() {
  if [[ "${SKIP_MLFLOW_PREFLIGHT}" == "1" ]]; then
    echo "Skipping MLflow preflight."
    return
  fi

  MLFLOW_HTTP_REQUEST_TIMEOUT="${MLFLOW_HTTP_REQUEST_TIMEOUT:-10}" \
  MLFLOW_HTTP_REQUEST_MAX_RETRIES="${MLFLOW_HTTP_REQUEST_MAX_RETRIES:-0}" \
    "${PYTHON_BIN}" - "${CONFIG_PATH}" <<'PY'
import os
import sys

import mlflow
import yaml
from mlflow import MlflowClient

with open(sys.argv[1]) as stream:
    config = yaml.safe_load(stream) or {}

mlflow_config = config.get("mlflow", {}) or {}
if not mlflow_config.get("enabled", False):
    print("MLflow is disabled; skipping tracker preflight.")
    raise SystemExit

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or mlflow_config.get(
    "tracking_uri"
)
if not tracking_uri:
    raise RuntimeError("MLflow is enabled but no tracking URI is configured")

mlflow.set_tracking_uri(tracking_uri)
MlflowClient().search_experiments(max_results=1)
print(f"Authenticated MLflow preflight succeeded: {tracking_uri}")
PY
}

launch_remote_node() {
  local host="$1"
  local node_rank="$2"
  local log_file="${LOG_DIR}/node_rank_${node_rank}_${host}.log"

  echo "Launching node rank ${node_rank} on ${host}; log: ${log_file}"
  "${SSH[@]}" "${host}" bash -s -- \
    "${PROJECT_DIR}" "${CONFIG_PATH}" "${PYTHON_BIN}" "${TRAIN_ENTRYPOINT}" \
    "${NNODES}" "${NPROC_PER_NODE}" "${node_rank}" \
    "${MASTER_ADDR}" "${MASTER_PORT}" "${NCCL_DEBUG_VALUE}" \
    "${MODEL_RUN_NAME}" "${NCCL_SOCKET_IFNAME_VALUE}" >"${log_file}" 2>&1 <<'REMOTE' &
set -Eeuo pipefail
project_dir="$1"
config_path="$2"
python_bin="$3"
train_entrypoint="$4"
nnodes="$5"
nproc_per_node="$6"
node_rank="$7"
master_addr="$8"
master_port="$9"
nccl_debug="${10}"
launch_run_name="${11}"
nccl_socket_ifname="${12:-}"

cd "${project_dir}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${nccl_debug}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PTYCHO_FM_LAUNCH_RUN_NAME="${launch_run_name}"
if [[ -n "${nccl_socket_ifname}" ]]; then
  export NCCL_SOCKET_IFNAME="${nccl_socket_ifname}"
fi

torchrun_pid=""
stop_torchrun() {
  if [[ -n "${torchrun_pid}" ]] && kill -0 "${torchrun_pid}" 2>/dev/null; then
    kill -TERM "${torchrun_pid}" 2>/dev/null || true
    wait "${torchrun_pid}" 2>/dev/null || true
  fi
}
trap 'stop_torchrun; exit 130' HUP INT TERM

"${python_bin}" -m torch.distributed.run \
  --nnodes "${nnodes}" \
  --nproc-per-node "${nproc_per_node}" \
  --node-rank "${node_rank}" \
  --master-addr "${master_addr}" \
  --master-port "${master_port}" \
  --max-restarts 0 \
  "${train_entrypoint}" --config "${config_path}" &
torchrun_pid=$!
set +e
wait "${torchrun_pid}"
status=$?
set -e
trap - HUP INT TERM
exit "${status}"
REMOTE
  PIDS+=("$!")
}

launch_local_node() {
  local log_file="${LOG_DIR}/node_rank_0_${LOCAL_HOST}.log"
  local -a command=(
    "${PYTHON_BIN}" -m torch.distributed.run
    --nnodes "${NNODES}"
    --nproc-per-node "${NPROC_PER_NODE}"
    --node-rank 0
    --master-addr "${MASTER_ADDR}"
    --master-port "${MASTER_PORT}"
    --max-restarts 0
    "${TRAIN_ENTRYPOINT}" --config "${CONFIG_PATH}"
  )

  echo "Launching node rank 0 on ${LOCAL_HOST}; log: ${log_file}"
  (
    cd "${PROJECT_DIR}"
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export PYTHONUNBUFFERED=1
    export NCCL_DEBUG="${NCCL_DEBUG_VALUE}"
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export PTYCHO_FM_LAUNCH_RUN_NAME="${MODEL_RUN_NAME}"
    if [[ -n "${NCCL_SOCKET_IFNAME_VALUE}" ]]; then
      export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME_VALUE}"
    fi
    exec "${command[@]}"
  ) >"${log_file}" 2>&1 &
  PIDS+=("$!")
}

PIDS=()

terminate_remote_torchrun() {
  local host="$1"
  local node_rank="$2"

  "${SSH[@]}" "${host}" bash -s -- \
    "${node_rank}" "${MASTER_ADDR}" "${MASTER_PORT}" <<'REMOTE' || true
set -Eeuo pipefail
node_rank="$1"
master_addr="$2"
master_port="$3"

pattern="[t]orch.distributed.run.*--node-rank ${node_rank}.*--master-addr ${master_addr}.*--master-port ${master_port}"
mapfile -t torchrun_pids < <(pgrep -f "${pattern}" || true)
if (( ${#torchrun_pids[@]} > 0 )); then
  kill -TERM "${torchrun_pids[@]}" 2>/dev/null || true
fi
REMOTE
}

terminate_launches() {
  local pid
  echo "Stopping launch processes..." >&2
  terminate_remote_torchrun "${REMOTE_HOST_1}" 1
  terminate_remote_torchrun "${REMOTE_HOST_2}" 2
  for pid in "${PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
}

trap 'terminate_launches; exit 130' INT TERM

echo "Preflight for ${NNODES} nodes x ${NPROC_PER_NODE} ranks:"
echo "  node rank 0: ${LOCAL_HOST}"
echo "  node rank 1: ${REMOTE_HOST_1}"
echo "  node rank 2: ${REMOTE_HOST_2}"
echo "  rendezvous:  ${MASTER_ADDR}:${MASTER_PORT}"
echo "  project:     ${PROJECT_DIR}"
echo "  config:      ${CONFIG_PATH}"
echo "  model run:   ${MODEL_RUN_NAME}"
echo "  launch:      ${RUN_STAMP}"
echo "  logs:        ${LOG_DIR}"

check_local_node
echo "  ${LOCAL_HOST}: at least ${NPROC_PER_NODE} GPUs and rendezvous port verified"
check_remote_node "${REMOTE_HOST_1}"
check_remote_node "${REMOTE_HOST_2}"
preflight_mlflow
stage_resume_checkpoint

# Start remote agents first; they wait for node rank 0 at the rendezvous.
launch_remote_node "${REMOTE_HOST_1}" 1
launch_remote_node "${REMOTE_HOST_2}" 2
launch_local_node

echo "All torchrun agents launched. Follow rank 0 with:"
echo "  tail -f '${LOG_DIR}/node_rank_0_${LOCAL_HOST}.log'"

remaining="${#PIDS[@]}"
while (( remaining > 0 )); do
  set +e
  wait -n
  status=$?
  set -e
  if (( status != 0 )); then
    echo "ERROR: A node launcher exited with status ${status}." >&2
    terminate_launches
    wait || true
    exit "${status}"
  fi
  remaining=$((remaining - 1))
done

echo "Training completed successfully on all ${NNODES} nodes."
