# src/copytrading_sim/paths.py

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

CONFIGS_DIR = PROJECT_ROOT / "configs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DATASETS_DIR = ARTIFACTS_DIR / "datasets"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
FIGURES_DIR = ARTIFACTS_DIR / "figures"


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist and return it as Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root if it is not absolute."""
    path = Path(path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path