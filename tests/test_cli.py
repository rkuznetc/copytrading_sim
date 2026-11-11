from copytrading_sim.datasets.exporters import load_run
from copytrading_sim.main import main


def test_cli_generates_saved_run(tmp_path, capsys):
    code = main(["--config", "configs/small.yaml", "--output-dir", str(tmp_path),
                 "--run-id", "smoke", "--override", "configs/logging_policy.yaml"])
    assert code == 0
    assert "Run saved:" in capsys.readouterr().out
    loaded = load_run(tmp_path / "smoke")
    assert len(loaded.tables["users"]) == 50
    assert len(loaded.tables["exposures"]) == 1250
    assert loaded.summary["copy_rate_per_slate"] <= 1


def test_cli_missing_config(capsys):
    assert main(["--config", "configs/not_exists.yaml"]) == 2
    assert "Config file not found" in capsys.readouterr().err


def test_cli_does_not_overwrite(tmp_path, capsys):
    args = ["--config", "configs/small.yaml", "--output-dir", str(tmp_path), "--run-id", "same"]
    assert main(args) == 0
    assert main(args) == 2
    assert "File exists" in capsys.readouterr().err
    load_run(tmp_path / "same")


def test_cli_negative_preview(capsys):
    assert main(["--preview", "-1"]) == 2
    assert "non-negative" in capsys.readouterr().err
