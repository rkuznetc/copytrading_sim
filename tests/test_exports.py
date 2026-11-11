import json

import numpy as np
import pandas as pd
import pytest

from copytrading_sim.datasets.exporters import load_run, save_run
from copytrading_sim.simulator.pipeline import run_pipeline


def test_roundtrip_preserves_values_and_dtypes(tiny_config, tmp_path):
    original = run_pipeline(tiny_config)
    path = save_run(original, tiny_config, output_dir=tmp_path, run_id="test")
    loaded = load_run(path)
    for name, table in original.items():
        pd.testing.assert_frame_equal(table, loaded.tables[name], check_exact=True)
    assert loaded.validation["passed"]
    assert loaded.config["exposure_policy"]["compatibility_weight"] == 0.65
    assert loaded.manifest["source"]["source_sha256"]
    assert loaded.manifest["environment"]["packages"]["numpy"]
    assert loaded.summary["n_slates"] == 12


def test_empty_events_roundtrip(tiny_config, tmp_path):
    tiny_config["copy_decision"] = {"no_copy_bias": 100, "noise_std": 0}
    tables = run_pipeline(tiny_config)
    path = save_run(tables, tiny_config, output_dir=tmp_path)
    loaded = load_run(path)
    assert loaded.tables["copy_events"].empty
    for name in tables:
        pd.testing.assert_frame_equal(tables[name], loaded.tables[name], check_exact=True)


def test_strings_that_look_like_missing_values_are_preserved(tiny_config, tmp_path):
    tiny_config["project"]["name"] = "NA"
    tiny_config["project"]["version"] = "null"
    tiny_config["exposure_policy"] = {"policy_name": "None"}
    tables = run_pipeline(tiny_config)
    loaded = load_run(save_run(tables, tiny_config, output_dir=tmp_path))
    pd.testing.assert_frame_equal(tables["metadata"], loaded.tables["metadata"], check_exact=True)
    assert loaded.tables["exposures"]["policy_name"].eq("None").all()


def test_same_data_same_digest_different_run_id(tiny_config, tmp_path):
    tables = run_pipeline(tiny_config)
    a = load_run(save_run(tables, tiny_config, tmp_path, "a"))
    b = load_run(save_run(tables, tiny_config, tmp_path, "b"))
    assert a.manifest["run_id"] != b.manifest["run_id"]
    assert a.manifest["dataset_sha256"] == b.manifest["dataset_sha256"]


def test_never_overwrite_existing_run(tiny_config, tmp_path):
    tables = run_pipeline(tiny_config)
    path = save_run(tables, tiny_config, tmp_path, "protected")
    before = (path / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        save_run(tables, tiny_config, tmp_path, "protected")
    assert (path / "manifest.json").read_bytes() == before
    load_run(path)


@pytest.mark.parametrize("name", ["../escape", "/tmp/escape", "", ".", "a/b", "x" * 97])
def test_unsafe_run_id_rejected(tiny_config, tmp_path, name):
    with pytest.raises(ValueError, match="run_id"):
        save_run(run_pipeline(tiny_config), tiny_config, tmp_path, name)
    assert not list(tmp_path.iterdir())


def test_modified_csv_detected(tiny_config, tmp_path):
    path = save_run(run_pipeline(tiny_config), tiny_config, tmp_path)
    with (path / "tables/users.csv").open("a") as stream:
        stream.write("broken\n")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_run(path)


def test_modified_config_detected(tiny_config, tmp_path):
    path = save_run(run_pipeline(tiny_config), tiny_config, tmp_path)
    with (path / "config.resolved.yaml").open("a") as stream:
        stream.write("# changed\n")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_run(path)


def test_manifest_path_traversal_rejected(tiny_config, tmp_path):
    path = save_run(run_pipeline(tiny_config), tiny_config, tmp_path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["tables"]["users"]["path"] = "../outside.csv"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unexpected table path"):
        load_run(path)


def test_failed_write_removes_only_new_directory(tiny_config, tmp_path, monkeypatch):
    tables = run_pipeline(tiny_config)
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("untouched")

    def fail(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail)
    with pytest.raises(OSError, match="simulated write failure"):
        save_run(tables, tiny_config, tmp_path, "failed")
    assert not (tmp_path / "failed").exists()
    assert sentinel.read_text() == "untouched"


def test_external_rng_saved_seed_reproduces_all_tables(tiny_config, tmp_path):
    original = run_pipeline(tiny_config, rng=np.random.default_rng(2026))
    loaded = load_run(save_run(original, tiny_config, tmp_path))
    assert loaded.config["random"]["seed"] == int(original["metadata"].iloc[0]["seed"])
    replay = run_pipeline(loaded.config)
    for name in original:
        pd.testing.assert_frame_equal(original[name], replay[name], check_exact=True)


def test_incomplete_directory_is_not_a_saved_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="completed run manifest"):
        load_run(tmp_path)
