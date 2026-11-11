"""Small provenance helpers; no Git repository or remote access is required."""

from __future__ import annotations

import hashlib
from importlib import metadata
import platform
from pathlib import Path
import subprocess
from typing import Any

RUNTIME_PACKAGES = ("numpy", "pandas", "PyYAML", "scikit-learn", "matplotlib")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_provenance() -> dict[str, Any]:
    """Record actual Python source hashes even in an unpacked, Git-free archive."""
    package_dir = Path(__file__).resolve().parent
    project_root = package_dir.parent.parent
    files = {
        "src/copytrading_sim/" + path.relative_to(package_dir).as_posix(): sha256_file(path)
        for path in sorted(package_dir.rglob("*.py"))
    }
    if (project_root / "pyproject.toml").is_file():
        files["pyproject.toml"] = sha256_file(project_root / "pyproject.toml")
    payload = "\n".join(f"{name}:{digest}" for name, digest in sorted(files.items()))
    result: dict[str, Any] = {
        "source_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "source_files": files,
        "git_commit": None,
        "git_dirty": None,
    }

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(project_root), *args],
            text=True, stderr=subprocess.DEVNULL, timeout=3,
        ).strip()

    try:
        # Do not report a parent repository unrelated to the installed package.
        if Path(git("rev-parse", "--show-toplevel")).resolve() == project_root.resolve():
            result["git_commit"] = git("rev-parse", "HEAD")
            result["git_dirty"] = bool(git("status", "--porcelain", "--untracked-files=normal"))
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def environment_provenance() -> dict[str, Any]:
    packages = {}
    for name in RUNTIME_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }
