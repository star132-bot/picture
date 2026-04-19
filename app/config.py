from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
CONFIG_PATH = DATA_DIR / 'config.json'
STATE_PATH = DATA_DIR / 'library.json'
THUMB_DIR = DATA_DIR / 'thumbs'

DEFAULT_CONFIG: dict[str, Any] = {
    'image_dir': str((PROJECT_ROOT / 'sample_images').resolve()),
    'recycle_bin_dir': str((PROJECT_ROOT / 'data' / 'recycle_bin').resolve()),
    'model_name': 'mobilenet',
    'blur_threshold': 60.0,
    'thumbnail_size': 520,
    'scan_batch_size': 24,
}


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    Path(DEFAULT_CONFIG['image_dir']).mkdir(parents=True, exist_ok=True)
    Path(DEFAULT_CONFIG['recycle_bin_dir']).mkdir(parents=True, exist_ok=True)


def load_config() -> dict[str, Any]:
    ensure_directories()
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()

    with CONFIG_PATH.open('r', encoding='utf-8') as fh:
        current = json.load(fh)

    # Migrate the old, overly aggressive blur default to a safer starting point.
    if current.get('blur_threshold') == 120.0:
        current['blur_threshold'] = 60.0

    merged = DEFAULT_CONFIG.copy()
    merged.update(current)
    return merged


def save_config(config: dict[str, Any]) -> dict[str, Any]:
    ensure_directories()
    merged = DEFAULT_CONFIG.copy()
    merged.update(config)
    with CONFIG_PATH.open('w', encoding='utf-8') as fh:
        json.dump(merged, fh, ensure_ascii=False, indent=2)
    return merged


def load_state() -> dict[str, Any]:
    ensure_directories()
    if not STATE_PATH.exists():
        return {'images': [], 'scanned_at': None, 'summary': {}}
    with STATE_PATH.open('r', encoding='utf-8') as fh:
        state = json.load(fh)

    images = state.get('images', [])
    if images and any('rank' not in image for image in images):
        for index, image in enumerate(images, start=1):
            image.setdefault('rank', index)
    return state


def save_state(state: dict[str, Any]) -> dict[str, Any]:
    ensure_directories()
    with STATE_PATH.open('w', encoding='utf-8') as fh:
        json.dump(state, fh, ensure_ascii=False, indent=2)
    return state
