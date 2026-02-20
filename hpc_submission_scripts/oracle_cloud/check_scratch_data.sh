#!/usr/bin/env bash
set -euo pipefail

# Fixed settings
DATA_DIR="/mnt/localdisk/scratch/simulated_data_cleanedProbe_2"

# Optional: set to a file with one hostname per line to bypass mgmt query
NODELIST_FILE=""

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

printf "Hosts (%s):\n" "${#HOST_ARR[@]}"
printf '  - %s\n' "${HOST_ARR[@]}"

echo

echo "Counting *.hdf5 under: ${DATA_DIR}"

echo "${HOSTS}" | xargs -n 1 -P 8 -I '{}' bash -c '
  host="$1"
  count=$(ssh "${host}" "bash -lc \"if [[ -d \"'"${DATA_DIR}"'\" ]]; then find \"'"${DATA_DIR}"'\" -maxdepth 1 -type f -name '*.hdf5' | wc -l; else echo 0; fi\"")
  echo "${host}: ${count}"
' _ '{}'
