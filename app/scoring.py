from __future__ import annotations

import hashlib
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import requests
import tensorflow as tf
import torch
from PIL import Image, ImageFile, ImageOps
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_resnet_v2 import (
    InceptionResNetV2,
    preprocess_input as inception_preprocess,
)
from tensorflow.keras.applications.mobilenet import (
    MobileNet,
    preprocess_input as mobilenet_preprocess,
)
from tensorflow.keras.layers import Dense, Dropout

from .config import MODELS_DIR

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    'mobilenet': {
        'label': 'NIMA MobileNet',
        'filename': 'mobilenet_weights.h5',
        'size': 224,
        'expected_bytes': 13159768,
        'download_url': 'https://github.com/titu1994/neural-image-assessment/releases/download/v0.2/mobilenet_weights.h5',
        'builder': 'mobilenet',
        'speed_note': '速度快，适合初筛。',
    },
    'inception': {
        'label': 'NIMA Inception ResNet V2',
        'filename': 'inception_resnet_weights.h5',
        'size': 224,
        'expected_bytes': 219071424,
        'download_url': 'https://github.com/titu1994/neural-image-assessment/releases/download/v0.5/inception_resnet_weights.h5',
        'builder': 'inception',
        'speed_note': '精度更高，适合终筛。',
    },
}

SUPPORTED_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'
}


def image_id_for_path(image_path: Path) -> str:
    return hashlib.sha1(str(image_path).encode('utf-8')).hexdigest()[:16]


@dataclass
class ModelBundle:
    model: Model
    preprocess: Callable[[np.ndarray], np.ndarray]
    image_size: int
    source: str


class NimaScorer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[str, ModelBundle] = {}

    @staticmethod
    def _weight_ready(weight_path: Path, expected_bytes: int) -> bool:
        return weight_path.exists() and weight_path.stat().st_size >= expected_bytes

    def runtime_info(self) -> dict[str, Any]:
        return {
            'mps_built': bool(torch.backends.mps.is_built()),
            'mps_available': bool(torch.backends.mps.is_available()),
            'tensorflow_gpus': [device.name for device in tf.config.list_physical_devices('GPU')],
        }

    def list_models(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for model_name, meta in MODEL_REGISTRY.items():
            weight_path = MODELS_DIR / meta['filename']
            weight_ready = self._weight_ready(weight_path, meta['expected_bytes'])
            entries.append({
                'name': model_name,
                'label': meta['label'],
                'download_url': meta['download_url'],
                'weight_path': str(weight_path),
                'weight_exists': weight_ready,
                'weight_size_mb': round(weight_path.stat().st_size / (1024 * 1024), 2) if weight_path.exists() else None,
                'speed_note': meta['speed_note'],
            })
        return entries

    def download_weights(self, model_name: str) -> dict[str, Any]:
        meta = MODEL_REGISTRY[model_name]
        weight_path = MODELS_DIR / meta['filename']
        tmp_path = weight_path.with_suffix('.download')
        response = requests.get(meta['download_url'], stream=True, timeout=30)
        response.raise_for_status()
        total = 0
        with tmp_path.open('wb') as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                fh.write(chunk)
                total += len(chunk)
        tmp_path.replace(weight_path)
        with self._lock:
            self._cache.pop(model_name, None)
        return {
            'model_name': model_name,
            'weight_path': str(weight_path),
            'downloaded_mb': round(total / (1024 * 1024), 2),
        }

    def score_images(
        self,
        image_paths: list[Path],
        model_name: str,
        blur_threshold: float,
        batch_size: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        metadata = self.inspect_images(image_paths, blur_threshold)
        bundle, warnings, source = self.prepare_scoring(model_name)
        images = self.score_metadata(metadata, batch_size, bundle, source)
        return images, {'warnings': warnings, 'score_source': source}

    def inspect_images(self, image_paths: list[Path], blur_threshold: float) -> list[dict[str, Any]]:
        return self._collect_metadata(image_paths, blur_threshold)

    def prepare_scoring(self, model_name: str) -> tuple[ModelBundle | None, list[str], str]:
        warnings: list[str] = []
        bundle = self._load_model(model_name, warnings)
        return bundle, warnings, bundle.source if bundle is not None else 'heuristic'

    def score_metadata(
        self,
        metadata: list[dict[str, Any]],
        batch_size: int,
        bundle: ModelBundle | None,
        source: str,
    ) -> list[dict[str, Any]]:
        if bundle is not None:
            predictions = self._predict_scores(bundle, metadata, batch_size)
        else:
            predictions = {item['id']: self._heuristic_distribution(item) for item in metadata}

        images: list[dict[str, Any]] = []
        for item in metadata:
            distribution = predictions[item['id']]
            mean_score = float(sum((index + 1) * value for index, value in enumerate(distribution)))
            image = dict(item)
            image['distribution'] = [round(float(value), 6) for value in distribution]
            image['score'] = round(mean_score, 4)
            image['score_source'] = source
            image['hidden_reason'] = 'blur' if image['is_blurry'] else None
            images.append(image)

        return self.rank_images(images)

    @staticmethod
    def rank_images(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ranked = sorted(images, key=lambda entry: (entry.get('score', 0.0), entry.get('blur_score', 0.0)), reverse=True)
        for index, image in enumerate(ranked, start=1):
            image['rank'] = index
        return ranked

    def _collect_metadata(self, image_paths: list[Path], blur_threshold: float) -> list[dict[str, Any]]:
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(image_paths)))) as pool:
            return list(pool.map(lambda path: self._inspect_image(path, blur_threshold), image_paths))

    def _inspect_image(self, image_path: Path, blur_threshold: float) -> dict[str, Any]:
        image_id = image_id_for_path(image_path)
        try:
            with Image.open(image_path) as img:
                image = ImageOps.exif_transpose(img.convert('RGB'))
                width, height = image.size
                rgb = np.array(image)
        except Exception as exc:
            return {
                'id': image_id,
                'path': str(image_path),
                'filename': image_path.name,
                'width': 0,
                'height': 0,
                'extension': image_path.suffix.lower(),
                'blur_score': 0.0,
                'is_blurry': False,
                'error': f'无法读取图片: {exc}',
                'brightness': 0.0,
                'contrast': 0.0,
                'saturation': 0.0,
            }

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        saturation = float(np.mean(hsv[..., 1]))

        return {
            'id': image_id,
            'path': str(image_path),
            'filename': image_path.name,
            'width': width,
            'height': height,
            'extension': image_path.suffix.lower(),
            'blur_score': round(blur_score, 3),
            'is_blurry': blur_score < blur_threshold,
            'error': None,
            'brightness': round(brightness, 3),
            'contrast': round(contrast, 3),
            'saturation': round(saturation, 3),
        }

    def _predict_scores(
        self,
        bundle: ModelBundle,
        metadata: list[dict[str, Any]],
        batch_size: int,
    ) -> dict[str, list[float]]:
        scored_items = [item for item in metadata if item['error'] is None]
        predictions: dict[str, list[float]] = {}
        for start in range(0, len(scored_items), batch_size):
            chunk = scored_items[start:start + batch_size]
            batch = np.stack([
                self._prepare_tensor(Path(item['path']), bundle.image_size, bundle.preprocess)
                for item in chunk
            ])
            output = bundle.model.predict(batch, verbose=0)
            for item, vector in zip(chunk, output):
                predictions[item['id']] = [float(value) for value in vector.tolist()]

        for item in metadata:
            if item['error'] is not None:
                predictions[item['id']] = self._heuristic_distribution(item)
        return predictions

    def _prepare_tensor(
        self,
        image_path: Path,
        image_size: int,
        preprocess: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        with Image.open(image_path) as img:
            image = ImageOps.exif_transpose(img.convert('RGB')).resize((image_size, image_size))
        array = np.asarray(image, dtype=np.float32)
        array = np.expand_dims(array, axis=0)
        return preprocess(array)[0]

    def _heuristic_distribution(self, item: dict[str, Any]) -> list[float]:
        sharpness = min(item['blur_score'] / 400.0, 1.0)
        contrast = min(item['contrast'] / 80.0, 1.0)
        saturation = min(item['saturation'] / 160.0, 1.0)
        exposure = 1.0 - min(abs(item['brightness'] - 128.0) / 128.0, 1.0)
        composite = 0.42 * sharpness + 0.24 * contrast + 0.17 * saturation + 0.17 * exposure
        score = 1.0 + composite * 9.0

        bins = []
        sigma = 1.2
        total = 0.0
        for bin_index in range(1, 11):
            weight = math.exp(-((bin_index - score) ** 2) / (2 * sigma ** 2))
            bins.append(weight)
            total += weight
        return [value / total for value in bins]

    def _load_model(self, model_name: str, warnings: list[str]) -> ModelBundle | None:
        meta = MODEL_REGISTRY[model_name]
        weight_path = MODELS_DIR / meta['filename']
        if not self._weight_ready(weight_path, meta['expected_bytes']):
            warnings.append(f'{model_name} 权重不存在或尚未下载完整，已切换到启发式评分。')
            return None

        with self._lock:
            if model_name in self._cache:
                return self._cache[model_name]

            try:
                bundle = self._build_bundle(model_name, weight_path)
            except Exception as exc:
                warnings.append(f'{model_name} 模型加载失败，已切换到启发式评分: {exc}')
                return None

            self._cache[model_name] = bundle
            return bundle

    def _build_bundle(self, model_name: str, weight_path: Path) -> ModelBundle:
        meta = MODEL_REGISTRY[model_name]
        if meta['builder'] == 'mobilenet':
            base_model = MobileNet(
                (meta['size'], meta['size'], 3),
                alpha=1,
                include_top=False,
                pooling='avg',
                weights=None,
            )
            preprocess = mobilenet_preprocess
        else:
            base_model = InceptionResNetV2(
                input_shape=(meta['size'], meta['size'], 3),
                include_top=False,
                pooling='avg',
                weights=None,
            )
            preprocess = inception_preprocess

        x = Dropout(0.75)(base_model.output)
        x = Dense(10, activation='softmax')(x)
        model = Model(base_model.input, x)
        model.load_weights(weight_path)
        return ModelBundle(
            model=model,
            preprocess=preprocess,
            image_size=meta['size'],
            source='nima',
        )


def list_image_files(image_dir: Path, recycle_bin_dir: Path) -> list[Path]:
    if not image_dir.exists():
        return []

    files: list[Path] = []
    recycle_resolved = recycle_bin_dir.resolve() if recycle_bin_dir.exists() else recycle_bin_dir
    for path in image_dir.rglob('*'):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            path_resolved = path.resolve()
        except FileNotFoundError:
            continue
        if recycle_bin_dir.exists() and recycle_resolved in path_resolved.parents:
            continue
        files.append(path)
    return sorted(files)
