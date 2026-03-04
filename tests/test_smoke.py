import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.transformer import build_model
from src.utils.config import load_config


def test_config_and_model_smoke():
    cfg = load_config('configs/small_debug.yaml')
    model = build_model(1000, cfg)
    assert model is not None
    assert cfg['seed'] == 123
