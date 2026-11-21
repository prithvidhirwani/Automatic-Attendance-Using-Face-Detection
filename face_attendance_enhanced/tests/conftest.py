import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app  # noqa: E402


@pytest.fixture()
def app(tmp_path):
    cfg = {
        "TESTING": True,
        "DATABASE_PATH": tmp_path / "t.db", "INSTANCE_DIR": tmp_path, "SECRET_KEY_FILE": tmp_path / "k.txt",
        "TRAIN_DIR": tmp_path / "train", "TEST_DIR": tmp_path / "test",
        "GROUND_TRUTH_CSV": tmp_path / "test" / "ground_truth.csv",
        "MODEL_DIR": tmp_path / "models", "MODEL_PATH": tmp_path / "models" / "m.yml",
        "LABEL_MAP_PATH": tmp_path / "models" / "m.json", "RESULTS_DIR": tmp_path / "results",
    }
    return create_app(cfg)


@pytest.fixture()
def client(app):
    return app.test_client()
