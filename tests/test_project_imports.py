# tests/test_project_imports.py

def test_package_imports():
    import copytrading_sim

    assert copytrading_sim is not None


def test_core_modules_import():
    import copytrading_sim.config
    import copytrading_sim.paths

    assert copytrading_sim.config is not None
    assert copytrading_sim.paths is not None