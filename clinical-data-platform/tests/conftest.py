import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def set_env(tmp_path_factory):
    # Ensure local config/data dirs exist for tests
    os.environ.setdefault("ENV", "local")
    Path("data/sample_raw").mkdir(parents=True, exist_ok=True)
    Path("data/sample_standardized").mkdir(parents=True, exist_ok=True)

