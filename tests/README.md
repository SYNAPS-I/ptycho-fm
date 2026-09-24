# Ptycho-FM test suite

The default suite covers data loading, model construction and inference,
training state, configuration, analysis, and command dispatch with compact
fixtures.

```bash
uv run pytest
```

Multi-process tests are marked `integration` and excluded from the default
run. Run them separately with:

```bash
uv run pytest -m integration
```

Run every collected test with:

```bash
uv run pytest -m ""
```

`test_pack_hdf5_consistency.py` is a standalone verification script for
site-specific packed datasets and is intentionally excluded from pytest
collection. Run it with explicit source and packed-data paths:

```bash
uv run python tests/test_pack_hdf5_consistency.py \
  --source /path/to/source \
  --packed /path/to/packed
```
