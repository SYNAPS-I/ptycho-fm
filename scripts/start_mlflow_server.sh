#!/usr/bin/env bash
set -euo pipefail

# Edit these defaults to choose the interface and port used by the server.
# Use 127.0.0.1 for local-only access, or 0.0.0.0 to listen on all interfaces.
MLFLOW_SERVER_HOST="${MLFLOW_SERVER_HOST:-127.0.0.1}"
MLFLOW_SERVER_PORT="${MLFLOW_SERVER_PORT:-5000}"

# Optional overrides. The defaults keep MLflow metadata and artifacts in the
# repository's ignored mlflow-data directory. MLFLOW_ALLOWED_HOSTS is useful
# when clients connect through a hostname not accepted by MLflow's defaults.
MLFLOW_ALLOWED_HOSTS="${MLFLOW_ALLOWED_HOSTS:-}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
mlflow_data_dir="${MLFLOW_DATA_DIR:-${repo_root}/mlflow-data}"
backend_store_uri="${MLFLOW_BACKEND_STORE_URI:-sqlite:///${mlflow_data_dir}/mlflow.db}"
artifacts_destination="${MLFLOW_ARTIFACTS_DESTINATION:-${mlflow_data_dir}/artifacts}"

usage() {
    cat <<EOF
Usage: $(basename "$0")

Starts an MLflow tracking server. Edit MLFLOW_SERVER_HOST and
MLFLOW_SERVER_PORT near the top of this script, or override them for one run:

  MLFLOW_SERVER_HOST=0.0.0.0 MLFLOW_SERVER_PORT=5001 $(basename "$0")

Optional environment variables:
  MLFLOW_DATA_DIR                 Metadata and artifact directory
  MLFLOW_BACKEND_STORE_URI        MLflow backend database URI
  MLFLOW_ARTIFACTS_DESTINATION    Artifact storage URI or path
  MLFLOW_ALLOWED_HOSTS            Comma-separated HTTP Host allowlist
EOF
}

case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
    "") ;;
    *)
        echo "Unknown argument: $1" >&2
        usage >&2
        exit 2
        ;;
esac

if ! [[ "${MLFLOW_SERVER_PORT}" =~ ^[0-9]+$ ]] \
    || (( MLFLOW_SERVER_PORT < 1 || MLFLOW_SERVER_PORT > 65535 )); then
    echo "MLFLOW_SERVER_PORT must be an integer from 1 to 65535." >&2
    exit 2
fi

mkdir -p "${mlflow_data_dir}"
if [[ "${artifacts_destination}" != *://* ]]; then
    mkdir -p "${artifacts_destination}"
fi

server_args=(
    server
    --host "${MLFLOW_SERVER_HOST}"
    --port "${MLFLOW_SERVER_PORT}"
    --backend-store-uri "${backend_store_uri}"
    --artifacts-destination "${artifacts_destination}"
)

if [[ -n "${MLFLOW_ALLOWED_HOSTS}" ]]; then
    server_args+=(--allowed-hosts "${MLFLOW_ALLOWED_HOSTS}")
fi

echo "Starting MLflow on http://${MLFLOW_SERVER_HOST}:${MLFLOW_SERVER_PORT}"
echo "Backend store: ${backend_store_uri}"
echo "Artifact store: ${artifacts_destination}"

if [[ "${MLFLOW_SERVER_HOST}" == "0.0.0.0" ]]; then
    echo "Note: clients must use this machine's reachable hostname or IP address."
    echo "Secure network access separately; the bind address does not add authentication."
fi

if command -v uv >/dev/null 2>&1; then
    exec uv run --frozen mlflow "${server_args[@]}"
elif command -v mlflow >/dev/null 2>&1; then
    exec mlflow "${server_args[@]}"
else
    echo "Neither uv nor mlflow is available. Install the project dependencies first." >&2
    exit 127
fi
