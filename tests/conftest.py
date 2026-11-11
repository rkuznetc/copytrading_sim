import pytest


@pytest.fixture
def tiny_config():
    return {
        "project": {"name": "test_simulation", "version": "0.1.0"},
        "random": {"seed": 42},
        "simulation": {
            "n_users": 4, "n_strategies": 6,
            "exposures_per_user": 3, "slate_size": 2,
        },
        "output": {"dir": "artifacts/tests", "format": "csv"},
    }
