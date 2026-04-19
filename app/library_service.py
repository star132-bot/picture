from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .config import THUMB_DIR, load_config, load_state, save_state
from .scoring import NimaScorer, list_image_files


class LibraryService:
    def __init__(self, scorer: Optional[NimaScorer] = None) -> None:
        self.scorer = scorer or NimaScorer()

    @staticmethod
    def summary(images: list[dict[str, Any]]) -> dict[str, Any]:
        scored = [image.get('score') for image in images if image.get('score') is not None]
        score_source = next((image.get('score_source') for image in images if image.get('score_source') and image.get('score_source') != 'pending'), None)
        return {
            'total': len(images),
            'visible': sum(1 for image in images if not image.get('is_blurry')),
            'hidden_blurry': sum(1 for image in images if image.get('is_blurry')),
            'top_score': max(scored, default=None),
            'score_source': score_source,
        }

    def scan_library(self, config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        current_config = config or load_config()
        image_dir = Path(current_config['image_dir']).expanduser().resolve()
        recycle_bin_dir = Path(current_config['recycle_bin_dir']).expanduser().resolve()
        files = list_image_files(image_dir, recycle_bin_dir)
        images, details = self.scorer.score_images(
            image_paths=files,
            model_name=current_config['model_name'],
            blur_threshold=float(current_config['blur_threshold']),
            batch_size=int(current_config['scan_batch_size']),
        )
        state = {
            'images': images,
            'summary': self.summary(images),
            'scanned_at': datetime.now().isoformat(timespec='seconds'),
            'scan_details': details,
        }
        save_state(state)
        return state

    def load_images(self, include_hidden: bool = False) -> dict[str, Any]:
        state = load_state()
        all_images = state.get('images', [])
        images = all_images if include_hidden else [image for image in all_images if not image.get('is_blurry')]
        return {
            'images': images,
            'summary': state.get('summary') or self.summary(all_images),
            'view_count': len(images),
            'scanned_at': state.get('scanned_at'),
            'scan_details': state.get('scan_details', {}),
        }

    def move_to_trash(self, image_id: str, config: Optional[dict[str, Any]] = None) -> str:
        current_config = config or load_config()
        recycle_bin_dir = Path(current_config['recycle_bin_dir']).expanduser().resolve()
        recycle_bin_dir.mkdir(parents=True, exist_ok=True)

        state = load_state()
        images = state.get('images', [])
        image = next((entry for entry in images if entry['id'] == image_id), None)
        if image is None:
            raise FileNotFoundError('图片不存在于当前索引中')

        source_path = Path(image['path'])
        if not source_path.exists():
            raise FileNotFoundError('源图片不存在')

        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        destination = recycle_bin_dir / f'{stamp}-{source_path.name}'
        counter = 1
        while destination.exists():
            destination = recycle_bin_dir / f'{stamp}-{counter}-{source_path.name}'
            counter += 1

        shutil.move(str(source_path), str(destination))
        state['images'] = [entry for entry in images if entry['id'] != image_id]
        state['summary'] = self.summary(state['images'])
        save_state(state)

        thumb_prefix = f'{image_id}_'
        for thumb in THUMB_DIR.glob(f'{thumb_prefix}*.jpg'):
            thumb.unlink(missing_ok=True)

        return str(destination)

    def empty_trash(self, config: Optional[dict[str, Any]] = None) -> int:
        current_config = config or load_config()
        recycle_bin_dir = Path(current_config['recycle_bin_dir']).expanduser().resolve()
        recycle_bin_dir.mkdir(parents=True, exist_ok=True)

        removed = 0
        for path in sorted(recycle_bin_dir.rglob('*'), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
                removed += 1
            elif path.is_dir() and path != recycle_bin_dir:
                try:
                    path.rmdir()
                except OSError:
                    pass
        return removed
