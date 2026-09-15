import h5py
import numpy as np

from scripts.create_index import FULL_COLUMNS, MINIMAL_COLUMNS, create_index


def _write_pair(directory):
    directory.mkdir(parents=True)
    dp_path = directory / "sample_dp.hdf5"
    para_path = directory / "sample_para.hdf5"
    with h5py.File(dp_path, "w") as f:
        f.create_dataset("dp", data=np.ones((3, 4, 5), dtype=np.float32))
    with h5py.File(para_path, "w") as f:
        object_dataset = f.create_dataset(
            "object", data=np.arange(42, dtype=np.float32).reshape(1, 6, 7)
        )
        object_dataset.attrs["pixel_height_m"] = 2.5e-9
        f.create_dataset("probe", data=np.ones((1, 2, 4, 5), dtype=np.complex64))


def test_create_minimal_and_full_indexes_with_relative_paths(tmp_path):
    _write_pair(tmp_path / "nested")

    minimal = create_index(tmp_path, index_type="minimal")
    assert list(minimal.columns) == MINIMAL_COLUMNS
    assert minimal.to_dict("records") == [
        {"dp_path": "nested/sample_dp.hdf5", "n_dps": 3}
    ]

    full_path = tmp_path / "full.csv"
    full = create_index(tmp_path, full_path, index_type="full")
    assert list(full.columns) == FULL_COLUMNS
    assert full.loc[0, "dp_path"] == "nested/sample_dp.hdf5"
    assert full.loc[0, "n_dps"] == 3
    assert full.loc[0, "n_incoherent_modes"] == 2
    assert full.loc[0, "object_height"] == 6
    assert full.loc[0, "object_width"] == 7
