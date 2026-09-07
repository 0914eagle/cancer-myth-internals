import glob

import pytest

from src.config import ensure_dir, load_config


def write(tmp_path, text, name="c.yaml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_a_placeholder_resolves_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("CANCER_MYTH_DATA_ROOT", "/data9/somebody")
    cfg = load_config(write(tmp_path, "paths:\n  result_dir: ${CANCER_MYTH_DATA_ROOT}/x/results\n"))
    assert cfg["paths"]["result_dir"] == "/data9/somebody/x/results"


def test_a_known_placeholder_has_a_default(tmp_path, monkeypatch):
    monkeypatch.delenv("CANCER_MYTH_DATA_ROOT", raising=False)
    cfg = load_config(write(tmp_path, "paths:\n  result_dir: ${CANCER_MYTH_DATA_ROOT}/x\n"))
    assert cfg["paths"]["result_dir"] == "/data1/heejae/x"


def test_an_unknown_placeholder_raises(tmp_path):
    with pytest.raises(KeyError, match="NO_SUCH_VAR"):
        load_config(write(tmp_path, "paths:\n  result_dir: ${NO_SUCH_VAR}/x\n"))


def test_base_config_is_overridden_per_section(tmp_path, monkeypatch):
    monkeypatch.setenv("CANCER_MYTH_DATA_ROOT", "/d")
    write(tmp_path, "a:\n  k: 1\nsource_model:\n  model_id: base\n  n_layers: 3\n", "base.yaml")
    cfg = load_config(write(tmp_path, "_base: base.yaml\nsource_model:\n  model_id: child\n  n_layers: 5\n"))
    assert cfg["source_model"] == {"model_id": "child", "n_layers": 5}
    assert cfg["a"] == {"k": 1}
    assert "_base" not in cfg


def test_the_shipped_configs_all_resolve(monkeypatch):
    monkeypatch.setenv("CANCER_MYTH_DATA_ROOT", "/data1/heejae")
    for path in sorted(glob.glob("configs/*.yaml")):
        cfg = load_config(path)
        assert "${" not in str(cfg["paths"]), path
        src = cfg["source_model"]
        assert src["device_map"] == "auto", path
        assert src.get("max_memory"), path
        assert all(str(v).endswith("GiB") for v in src["max_memory"].values()), path
        assert int(src["n_layers"]) > 0 and int(src["d_model"]) > 0, path


def test_a_root_path_from_an_empty_variable_names_the_unsourced_shell(monkeypatch):
    import pathlib

    def refuse(self, *a, **k):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(pathlib.Path, "mkdir", refuse)
    with pytest.raises(PermissionError) as excinfo:
        ensure_dir("/results")
    assert "source scripts/env.sh" in str(excinfo.value)
