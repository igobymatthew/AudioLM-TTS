import argparse
import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _set_nested(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split('.')
    cur = cfg
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def _parse_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except Exception:
        return raw


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}

    base = cfg.get('base_config')
    if base:
        parent = load_config(base)
        cfg = _deep_merge(parent, {k: v for k, v in cfg.items() if k != 'base_config'})

    for item in overrides or []:
        if '=' not in item:
            raise ValueError(f'Override must be key=value, got: {item}')
        key, value = item.split('=', 1)
        _set_nested(cfg, key, _parse_value(value))
    return cfg


def build_config_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('--set', action='append', default=[], help='Override config entries as key=value')
    return parser
