#!/usr/bin/env bash
set -euo pipefail

# Fixed paths per your assumptions
# Do not add slash to the end of SRC if you want to copy the folder as is
SRC="/fss/datasets/synaps-i/simulated_data_cleanedProbe"
DEST_BASE="/mnt/localdisk/scratch"
DEST="${DEST_BASE}/simulated_data_cleanedProbe"

WORKERS=6
DRY_RUN=0
NODELIST_FILE="/fss/projects/synaps-i/mingdu/ptycho-vit/workspace/node_list.txt"

usage() {
  cat <<EOF
Usage:
  $0 [--workers N] [--dry-run] [--nodelist FILE]

What it does:
  - Queries compute nodes via: mgmt nodes list --fields role=compute --format json
    (or reads from --nodelist FILE if provided)
  - Extracts hostnames (compute role + active/ready/running when available)
  - Copies ${SRC} to ${DEST} on each node
  - Runs in parallel across nodes with --workers (default: ${WORKERS})

Example:
  $0 --workers 16
  $0 --nodelist /path/to/nodes.txt
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers) WORKERS="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --nodelist) NODELIST_FILE="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 2 ;;
  esac
done

if ! [[ "${WORKERS}" =~ ^[0-9]+$ ]] || [[ "${WORKERS}" -lt 1 ]]; then
  echo "ERROR: --workers must be a positive integer"
  exit 2
fi

if [[ ! -d "${SRC}" ]]; then
  echo "ERROR: Source directory not found: ${SRC}"
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
import json, os, sys

try:
    raw = os.environ["NODES_JSON"]
except KeyError:
    raise RuntimeError('NODES_JSON is not set (script should set it via mgmt).')

data = json.loads(raw)

# Support either a list (your sample) or {"items":[...]}
items = data.get("items") if isinstance(data, dict) else data
if not isinstance(items, list):
    raise RuntimeError(f"Unexpected NODES_JSON shape: {type(items).__name__}")

def get_hostname(n: dict) -> str | None:
    return n.get("hostname") or (n.get("fields") or {}).get("hostname")

def get_status(n: dict) -> str:
    return (n.get("status") or (n.get("fields") or {}).get("status") or "").lower()

# Prefer "healthy/runnable" statuses when available
preferred = []
for n in items:
    hn = get_hostname(n)
    if not hn:
        continue
    st = get_status(n)
    if st and st not in ("active", "ready", "running"):
        continue
    preferred.append(hn)

# If status-filtering produced nothing, fall back to "all hostnames"
hosts = preferred
if not hosts:
    hosts = [hn for hn in (get_hostname(n) for n in items) if hn]

print("\n".join(hosts))
PY
<<< "${NODES_JSON}")"
fi

if [[ -z "${HOSTS}" ]]; then
  echo "ERROR: No hosts found"
  exit 1
fi

echo
echo "Staging:"
echo "  SRC     : ${SRC}"
echo "  DEST    : ${DEST}"
echo "  WORKERS : ${WORKERS}"
echo "  DRY RUN : ${DRY_RUN}"
echo
echo "Hosts:"
echo "${HOSTS}" | sed 's/^/  - /'
echo

stage_one() {
  local host="$1"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[DRY] ssh ${host} mkdir -p '${DEST_BASE}'"
    echo "[DRY] rsync -az '${SRC}' ${host}:'${DEST_BASE}/'"
    return 0
  fi

  ssh "${host}" "mkdir -p '${DEST_BASE}'"
  echo "---- ${host} ----"
  # Copy the directory itself into DEST_BASE, yielding DEST_BASE/simulated_data
  rsync -az --partial --info=stats2,progress2 "${SRC}" "${host}:${DEST_BASE}/"
  echo "OK   ${host}"
}

export -f stage_one
export SRC DEST_BASE DEST DRY_RUN

echo "${HOSTS}" | xargs -n 1 -P "${WORKERS}" -I '{}' \
  bash -c 'stage_one "$@"' _ '{}'

echo
echo "Done."
