"""Persist complete simulation runs and load them without rerunning the simulator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any
from uuid import uuid4

import pandas as pd
import yaml

from copytrading_sim.config import resolve_config
from copytrading_sim.paths import resolve_project_path
from copytrading_sim.provenance import environment_provenance, sha256_file, source_provenance
from copytrading_sim.randomness import RNG_SCHEME, STREAM_IDS
from copytrading_sim.validation.structural_checks import TABLE_NAMES, validate_tables

RUN_FORMAT_VERSION = 1
SIDECAR_FILES = ("config.resolved.yaml", "summary.json", "validation.json", "requirements.runtime.txt")
CSV_DTYPES = {"int64", "int32", "uint64", "float64", "float32", "object", "bool", "str", "string"}


@dataclass
class SavedRun:
    """Loaded tables plus their actual configuration and run provenance."""

    tables: dict[str, pd.DataFrame]
    config: dict[str, Any]
    manifest: dict[str, Any]
    summary: dict[str, Any]
    validation: dict[str, Any]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _dataset_digest(table_manifest: dict[str, Any]) -> str:
    payload = "\n".join(f"{name}:{table_manifest[name]['sha256']}" for name in TABLE_NAMES)
    return hashlib.sha256(payload.encode()).hexdigest()


def build_summary(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Descriptive counts only; this is not model evaluation or realism validation."""
    decisions = tables["copy_decisions"]
    copied_by_slate = decisions.groupby("slate_id")["copied"].sum()
    return {
        "table_rows": {name: len(table) for name, table in tables.items()},
        "n_slates": int(len(copied_by_slate)),
        "n_copied_slates": int(copied_by_slate.gt(0).sum()),
        "n_no_copy_slates": int(copied_by_slate.eq(0).sum()),
        "copy_rate_per_exposed_item": float(decisions["copied"].mean()),
        "copy_rate_per_slate": float(copied_by_slate.gt(0).mean()),
        "user_types": {str(k): int(v) for k, v in tables["users"]["user_type"].value_counts().items()},
        "strategy_types": {
            str(k): int(v) for k, v in tables["strategies"]["strategy_type"].value_counts().items()
        },
    }


def save_run(
    tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    run_id: str | None = None,
) -> Path:
    """Validate and save an exclusive run directory; never overwrite an existing run.

    The manifest is written last, so an interrupted run is not loadable as complete.
    An external RNG passed to run_pipeline is represented by its recorded effective
    root seed. No dataset is regenerated during export.
    """
    resolved = resolve_config(config)
    if "metadata" not in tables or len(tables["metadata"]) != 1:
        raise ValueError("Expected exactly one metadata row before saving")
    seed = tables["metadata"].iloc[0]["seed"]
    # resolve_config also rejects non-integral or negative metadata seeds.
    resolved["random"]["seed"] = seed
    if output_dir is not None:
        resolved["output"]["dir"] = str(output_dir)
    resolved = resolve_config(resolved)
    validation = validate_tables(tables, resolved)
    summary = build_summary(tables)

    now = datetime.now(timezone.utc)
    if run_id is None:
        run_id = f"{now:%Y%m%dT%H%M%S%fZ}-s{resolved['random']['seed']}-{uuid4().hex[:8]}"
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", run_id):
        raise ValueError("run_id must contain 1..96 ASCII letters/digits, '.', '_' or '-'")
    root = resolve_project_path(Path(resolved["output"]["dir"]).expanduser()).resolve()
    run_dir = root / run_id
    # Exclusive creation also protects against two processes using the same run_id.
    run_dir.mkdir(parents=True, exist_ok=False)
    completed = False
    try:
        (run_dir / "tables").mkdir()
        table_manifest = {}
        for name in TABLE_NAMES:
            frame = tables[name]
            dtypes = {col: str(dtype) for col, dtype in frame.dtypes.items()}
            if not set(dtypes.values()).issubset(CSV_DTYPES):
                raise ValueError(f"Unsupported CSV dtypes in {name}: {dtypes}")
            relative = f"tables/{name}.csv"
            path = run_dir / relative
            frame.to_csv(path, index=False, encoding="utf-8", float_format="%.17g", lineterminator="\n")
            table_manifest[name] = {
                "path": relative, "rows": len(frame), "columns": list(frame.columns),
                "dtypes": dtypes, "sha256": sha256_file(path),
            }

        (run_dir / "config.resolved.yaml").write_text(
            yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        _write_json(run_dir / "summary.json", summary)
        _write_json(run_dir / "validation.json", validation)
        environment = environment_provenance()
        requirements = [f"# Runtime package versions; Python {environment['python']}."]
        requirements += [
            f"{name}=={version}" for name, version in environment["packages"].items() if version
        ]
        (run_dir / "requirements.runtime.txt").write_text("\n".join(requirements) + "\n")
        manifest = {
            "run_format_version": RUN_FORMAT_VERSION,
            "run_id": run_id,
            "created_at_utc": now.isoformat(),
            "dataset_sha256": _dataset_digest(table_manifest),
            "generation_config_sha256": str(tables["metadata"].iloc[0]["generation_config_sha256"]),
            "random": {
                "root_seed": resolved["random"]["seed"],
                "scheme": RNG_SCHEME, "bit_generator": "PCG64", "stream_ids": STREAM_IDS,
            },
            "source": source_provenance(),
            "environment": environment,
            "tables": table_manifest,
            "files": {name: sha256_file(run_dir / name) for name in SIDECAR_FILES},
        }
        _write_json(run_dir / "manifest.json", manifest)
        completed = True
    finally:
        if not completed:
            # We own this freshly created directory; never remove a pre-existing run.
            shutil.rmtree(run_dir, ignore_errors=True)
    return run_dir


def load_run(run_dir: str | Path) -> SavedRun:
    """Verify hashes/schema, restore CSV dtypes, and recheck structural invariants.

    Paths are relative to the caller's working directory. Unlike generation config
    paths, saved-run paths can point anywhere, including outside the repository.
    Checksums detect accidental modifications; they are not a digital signature.
    """
    root = Path(run_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No completed run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("run_format_version") != RUN_FORMAT_VERSION:
        raise ValueError("Unsupported or malformed run manifest")
    if set(manifest.get("tables", {})) != set(TABLE_NAMES):
        raise ValueError("Manifest must describe exactly the seven simulation tables")
    if set(manifest.get("files", {})) != set(SIDECAR_FILES):
        raise ValueError("Manifest sidecar files are missing or unexpected")
    for name in SIDECAR_FILES:
        if sha256_file(root / name) != manifest["files"][name]:
            raise ValueError(f"Checksum mismatch: {name}")
    resolved = resolve_config(yaml.safe_load((root / "config.resolved.yaml").read_text(encoding="utf-8")))
    tables = {}
    for name in TABLE_NAMES:
        info = manifest["tables"][name]
        relative = f"tables/{name}.csv"
        if info.get("path") != relative:
            raise ValueError(f"Unexpected table path for {name}")
        path = root / relative
        if sha256_file(path) != info["sha256"]:
            raise ValueError(f"Checksum mismatch: {relative}")
        dtypes = info["dtypes"]
        if set(dtypes) != set(info["columns"]) or not set(dtypes.values()).issubset(CSV_DTYPES):
            raise ValueError(f"Invalid dtype schema: {name}")
        frame = pd.read_csv(
            path, dtype=dtypes, encoding="utf-8", keep_default_na=False, float_precision="round_trip",
        )
        if len(frame) != info["rows"] or list(frame.columns) != info["columns"]:
            raise ValueError(f"Schema or row-count mismatch: {name}")
        tables[name] = frame
    if _dataset_digest(manifest["tables"]) != manifest["dataset_sha256"]:
        raise ValueError("Dataset checksum mismatch")
    validation = validate_tables(tables, resolved)
    summary = build_summary(tables)
    saved_summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    saved_validation = json.loads((root / "validation.json").read_text(encoding="utf-8"))
    if summary != saved_summary or not saved_validation.get("passed"):
        raise ValueError("Saved summary or validation status is inconsistent")
    if manifest["generation_config_sha256"] != tables["metadata"].iloc[0]["generation_config_sha256"]:
        raise ValueError("Manifest configuration digest is inconsistent")
    return SavedRun(tables=tables, config=resolved, manifest=manifest,
                    summary=summary, validation=validation)
