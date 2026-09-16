# S26 fly-scan inference

Run a single scan with `run_flyscan_inference.py`. To distribute complete
scans across GPUs, invoke `flyscan_inference_dispatch.py` once per node.
It waits for GPUs with no active compute processes, pins each child to one GPU
(`cuda:0` inside the child), and reports a nonzero exit status if a worker fails.
The launcher does not allocate nodes or start remote processes itself.

Single-node example:

```bash
python scripts/s26/flyscan_inference_dispatch.py /data/scans \
  --node-rank 0 --num-nodes 1 --gpus 0 1 \
  --data-format raw-h5 --raw-crop 0 511 0 511 --bin 2 \
  --config config.yaml --checkpoint model.pth --output-dir results \
  --ny 201 --nx 200 --step-size 14 --scan-pattern zigzag
```

The directory must contain raw `.h5`/`.hdf5` files or paired
`*_dp.hdf5`/`*_para.hdf5` files directly (no recursive discovery).
Select the format explicitly with `--data-format`. Normalization uses an
explicit `--normalization-file` pickle mapping, or each file's maximum;
for raw data, maxima are computed after cropping and binning.
Scalar normalization belongs to the single-file CLI.
`--no-flip-patch-y`, `--no-apply-noise`, and crop sweeps are forwarded.
Output filenames and crop subdirectories follow the single-GPU script.

For multiple nodes, create a canonical plan once:

```bash
python scripts/s26/flyscan_inference_dispatch.py /data/scans \
  --node-rank 0 --num-nodes 2 --gpus 0 1 --data-format raw-h5 \
  --write-shard-plan plan.json --plan-only
```

Then run the normal command on each node with `--num-nodes 2`,
`--shard-plan plan.json`, and a distinct `--node-rank` (0 or 1).
Use the same GPU count per node and the identical plan. Input roots may differ;
all planned filenames and byte sizes must match on every node. Validation uses
file sizes, not content hashes. This also works when the input root is shared.

`--dry-run` prints local assignments without creating temporary directories
or querying GPUs. Workers receive temporary symlink directories.
Their shell launch records are saved under `OUTPUT/launches/rankNNNNN.sh`;
use `--keep-shards` if these records need to be rerunnable, since normal
cleanup removes the temporary inputs. Original data is never removed.
Interruptions and launch errors terminate and reap children before cleanup.
