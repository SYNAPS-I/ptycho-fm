"""Config lookup for the installed training entry point."""

import sys

import pytest

from ptycho_fm import train


@pytest.mark.parametrize("config_option", ["default", "relative", "absolute"])
def test_training_reads_config_from_working_directory(tmp_path, monkeypatch, config_option):
    config_path = tmp_path / ("config.yaml" if config_option == "default" else "custom.yaml")
    config_path.write_text("marker: selected config\n")
    args = ["ptycho-fm-train"]
    if config_option != "default":
        args += ["--config", str(config_path) if config_option == "absolute" else config_path.name]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", args)

    class ConfigRead(Exception):
        pass

    def read_config(stream):
        assert stream.name == str(config_path)
        assert stream.read() == "marker: selected config\n"
        # Stop before distributed initialization or training.
        raise ConfigRead

    monkeypatch.setattr(train.yaml, "safe_load", read_config)
    with pytest.raises(ConfigRead):
        train.main()
