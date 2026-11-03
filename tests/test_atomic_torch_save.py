"""Checkpoint replacement remains complete when saves overlap or fail."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import torch

from ptycho_fm import training


def test_overlapping_saves_use_independent_temporary_files(tmp_path, monkeypatch):
    destination = tmp_path / "best_model.pth"
    torch.save({"value": torch.tensor([-1])}, destination)
    ready = Barrier(2)
    replace = training.os.replace

    def synchronized_replace(source, target):
        # Both writers finish serialization before either replaces the checkpoint.
        assert torch.load(destination, weights_only=True)["value"].item() == -1
        ready.wait(timeout=10)
        replace(source, target)

    monkeypatch.setattr(training.os, "replace", synchronized_replace)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(training._atomic_torch_save, {"value": torch.full((32,), value)}, destination)
            for value in (1, 2)
        ]
        for future in futures:
            future.result(timeout=15)

    saved = torch.load(destination, weights_only=True)["value"]
    assert torch.equal(saved, torch.full((32,), 1)) or torch.equal(saved, torch.full((32,), 2))
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("failure_stage", ["save", "replace"])
def test_failed_save_preserves_checkpoint_and_cleans_own_temporary_file(tmp_path, monkeypatch, failure_stage):
    destination = tmp_path / "best_model.pth"
    original = {"value": torch.tensor([1])}
    torch.save(original, destination)
    unrelated = tmp_path / "another_writer.tmp"
    unrelated.write_bytes(b"another writer's data")
    save = torch.save

    def fail_save(obj, path):
        save(obj, path)
        raise OSError("simulated serialization failure")

    def fail_replace(source, target):
        raise OSError("simulated replacement failure")

    if failure_stage == "save":
        monkeypatch.setattr(training.torch, "save", fail_save)
    else:
        monkeypatch.setattr(training.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        training._atomic_torch_save({"value": torch.tensor([2])}, destination)

    torch.testing.assert_close(torch.load(destination, weights_only=True)["value"], original["value"])
    assert set(tmp_path.iterdir()) == {destination, unrelated}
    assert unrelated.read_bytes() == b"another writer's data"
