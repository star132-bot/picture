from __future__ import annotations

import platform
import subprocess
import shutil
import threading
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from PIL import Image, ImageFile, ImageOps
from pydantic import BaseModel, Field

from .config import THUMB_DIR, ensure_directories, load_config, load_state, save_config, save_state
from .scoring import MODEL_REGISTRY, NimaScorer, image_id_for_path, list_image_files

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

app = FastAPI(title='Photo Ranker')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
app.mount('/static', StaticFiles(directory=Path(__file__).parent / 'static'), name='static')
templates = Jinja2Templates(directory=str(Path(__file__).parent / 'templates'))
scorer = NimaScorer()
scan_jobs: dict[str, dict[str, Any]] = {}
scan_jobs_lock = threading.Lock()


class ConfigPayload(BaseModel):
    image_dir: str
    recycle_bin_dir: str
    model_name: str = Field(pattern='^(mobilenet|inception)$')
    blur_threshold: float = Field(default=120.0, ge=0)
    thumbnail_size: int = Field(default=520, ge=200, le=1600)
    scan_batch_size: int = Field(default=24, ge=1, le=128)


class ModelDownloadPayload(BaseModel):
    model_name: str = Field(pattern='^(mobilenet|inception)$')


ensure_directories()


def _image_index() -> dict[str, dict[str, Any]]:
    return {image['id']: image for image in load_state().get('images', [])}


def _scan_image_index() -> dict[str, dict[str, Any]]:
    images: dict[str, dict[str, Any]] = {}
    with scan_jobs_lock:
        for job in scan_jobs.values():
            for image in job.get('images', []):
                images[image['id']] = image
    return images


def _summary(images: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [image.get('score') for image in images if image.get('score') is not None]
    score_source = next((image.get('score_source') for image in images if image.get('score_source') and image.get('score_source') != 'pending'), None)
    return {
        'total': len(images),
        'visible': sum(1 for image in images if not image.get('is_blurry')),
        'hidden_blurry': sum(1 for image in images if image.get('is_blurry')),
        'top_score': max(scored, default=None),
        'score_source': score_source,
    }


def _serialize_state() -> dict[str, Any]:
    config = load_config()
    state = load_state()
    return {
        'config': config,
        'state': state,
        'models': scorer.list_models(),
        'runtime': scorer.runtime_info(),
    }


def _placeholder_images(files: list[Path]) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    for index, path in enumerate(files, start=1):
        images.append({
            'id': image_id_for_path(path),
            'path': str(path),
            'filename': path.name,
            'width': 0,
            'height': 0,
            'extension': path.suffix.lower(),
            'blur_score': None,
            'is_blurry': False,
            'error': None,
            'brightness': 0.0,
            'contrast': 0.0,
            'saturation': 0.0,
            'distribution': [],
            'score': None,
            'score_source': 'pending',
            'hidden_reason': None,
            'rank': index,
            'scan_index': index,
        })
    return images


def _snapshot_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        'scan_id': job['scan_id'],
        'processed': job['processed'],
        'total': job['total'],
        'done': job['done'],
        'error': job['error'],
        'scanned_at': job.get('scanned_at'),
        'summary': dict(job['summary']),
        'scan_details': dict(job['scan_details']),
        'images': [dict(image) for image in job['images']],
    }


def _run_scan_job(scan_id: str, config: dict[str, Any], files: list[Path]) -> None:
    batch_size = max(1, int(config['scan_batch_size']))
    blur_threshold = float(config['blur_threshold'])
    bundle, warnings, source = scorer.prepare_scoring(config['model_name'])

    with scan_jobs_lock:
        job = scan_jobs[scan_id]
        images = job['images']

    image_index = {image['id']: image for image in images}
    total = len(files)

    try:
        for start in range(0, total, batch_size):
            batch_paths = files[start:start + batch_size]
            metadata = scorer.inspect_images(batch_paths, blur_threshold)
            scored_batch = scorer.score_metadata(metadata, batch_size, bundle, source)
            for scored_image in scored_batch:
                image_index[scored_image['id']].update(scored_image)
            processed = min(start + len(batch_paths), total)
            with scan_jobs_lock:
                current = scan_jobs.get(scan_id)
                if current is None:
                    return
                current['processed'] = processed
                current['summary'] = _summary(images)
                current['scan_details'] = {
                    'warnings': warnings,
                    'score_source': source,
                }

        ranked_images = scorer.rank_images([dict(image) for image in images])
        state = {
            'images': ranked_images,
            'summary': _summary(ranked_images),
            'scanned_at': datetime.now().isoformat(timespec='seconds'),
            'scan_details': {
                'warnings': warnings,
                'score_source': source,
            },
        }
        save_state(state)
        with scan_jobs_lock:
            current = scan_jobs.get(scan_id)
            if current is None:
                return
            current['images'] = ranked_images
            current['processed'] = total
            current['done'] = True
            current['summary'] = state['summary']
            current['scan_details'] = state['scan_details']
            current['scanned_at'] = state['scanned_at']
    except Exception as exc:
        with scan_jobs_lock:
            current = scan_jobs.get(scan_id)
            if current is None:
                return
            current['error'] = str(exc)
            current['done'] = True


def _placeholder_image(max_size: int) -> bytes:
    canvas = Image.new('RGB', (max_size, max_size), '#ddd4c7')
    buffer = BytesIO()
    canvas.save(buffer, format='JPEG', quality=88)
    return buffer.getvalue()


def _choose_folder_dialog(initial_path: Optional[str] = None) -> Optional[str]:
    if platform.system() != 'Darwin':
        raise HTTPException(status_code=501, detail='当前只实现了 macOS 文件夹选择器。')

    start_path = initial_path or str(Path.home())
    start_path = start_path.replace('\\', '\\\\').replace('"', '\\"')
    script = f'''
set defaultLocation to POSIX file "{start_path}"
try
  set chosenFolder to choose folder with prompt "请选择文件夹" default location defaultLocation
  POSIX path of chosenFolder
on error number -128
  return ""
end try
'''.strip()

    result = subprocess.run(
        ['osascript', '-e', script],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr.strip() or '文件夹选择器启动失败')

    selected = result.stdout.strip()
    return selected or None


@app.get('/', response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse('index.html', {'request': request})


@app.get('/api/status')
def status() -> dict[str, Any]:
    return _serialize_state()


@app.get('/api/models')
def models() -> dict[str, Any]:
    return {'models': scorer.list_models(), 'runtime': scorer.runtime_info()}


@app.post('/api/models/download')
def download_model(payload: ModelDownloadPayload) -> dict[str, Any]:
    result = scorer.download_weights(payload.model_name)
    return {'ok': True, 'result': result, 'models': scorer.list_models()}


@app.get('/api/config')
def get_config() -> dict[str, Any]:
    return load_config()


@app.get('/api/dialog/folder')
def choose_folder(initial_path: Optional[str] = Query(default=None)) -> dict[str, Any]:
    selected_path = _choose_folder_dialog(initial_path)
    return {'ok': True, 'path': selected_path}


@app.post('/api/config')
def update_config(payload: ConfigPayload) -> dict[str, Any]:
    config = save_config(payload.model_dump())
    Path(config['image_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['recycle_bin_dir']).mkdir(parents=True, exist_ok=True)
    return {'ok': True, 'config': config}


@app.post('/api/scan/start')
def start_scan() -> dict[str, Any]:
    config = load_config()
    image_dir = Path(config['image_dir']).expanduser().resolve()
    recycle_bin_dir = Path(config['recycle_bin_dir']).expanduser().resolve()
    files = list_image_files(image_dir, recycle_bin_dir)
    scan_id = uuid.uuid4().hex
    images = _placeholder_images(files)
    job = {
        'scan_id': scan_id,
        'images': images,
        'processed': 0,
        'total': len(files),
        'done': False,
        'error': None,
        'summary': _summary(images),
        'scan_details': {'warnings': [], 'score_source': None},
        'scanned_at': None,
    }
    with scan_jobs_lock:
        scan_jobs[scan_id] = job

    threading.Thread(target=_run_scan_job, args=(scan_id, config, files), daemon=True).start()
    return {'ok': True, **_snapshot_job(job)}


@app.get('/api/scan/{scan_id}')
def scan_status(scan_id: str) -> dict[str, Any]:
    with scan_jobs_lock:
        job = scan_jobs.get(scan_id)
        if job is None:
            raise HTTPException(status_code=404, detail='扫描任务不存在')
        return {'ok': True, **_snapshot_job(job)}


@app.post('/api/scan')
def scan_library() -> dict[str, Any]:
    config = load_config()
    image_dir = Path(config['image_dir']).expanduser().resolve()
    recycle_bin_dir = Path(config['recycle_bin_dir']).expanduser().resolve()

    files = list_image_files(image_dir, recycle_bin_dir)
    images, details = scorer.score_images(
        image_paths=files,
        model_name=config['model_name'],
        blur_threshold=float(config['blur_threshold']),
        batch_size=int(config['scan_batch_size']),
    )
    state = {
        'images': images,
        'summary': _summary(images),
        'scanned_at': datetime.now().isoformat(timespec='seconds'),
        'scan_details': details,
    }
    save_state(state)
    return {'ok': True, **_serialize_state()}


@app.get('/api/images')
def images(include_hidden: bool = Query(default=False)) -> dict[str, Any]:
    state = load_state()
    all_images = state.get('images', [])
    images = all_images
    if not include_hidden:
        images = [image for image in images if not image.get('is_blurry')]
    return {
        'images': images,
        'summary': state.get('summary') or _summary(all_images),
        'view_count': len(images),
        'scanned_at': state.get('scanned_at'),
        'scan_details': state.get('scan_details', {}),
    }


@app.get('/api/media/{image_id}')
def media(image_id: str, variant: str = Query(default='full'), max_size: int = Query(default=640, ge=128, le=2048)):
    image = _image_index().get(image_id) or _scan_image_index().get(image_id)
    if not image:
        raise HTTPException(status_code=404, detail='图片不存在')

    image_path = Path(image['path'])
    if not image_path.exists():
        raise HTTPException(status_code=404, detail='图片文件不存在')

    if image.get('error'):
        payload = _placeholder_image(max_size if variant != 'full' else max(max_size, 1200))
        return StreamingResponse(BytesIO(payload), media_type='image/jpeg')

    if variant == 'full':
        return FileResponse(image_path)

    thumb_path = THUMB_DIR / f'{image_id}_{max_size}.jpg'
    if thumb_path.exists() and thumb_path.stat().st_mtime >= image_path.stat().st_mtime:
        return FileResponse(thumb_path, media_type='image/jpeg')

    try:
        with Image.open(image_path) as img:
            frame = ImageOps.exif_transpose(img.convert('RGB'))
            frame.thumbnail((max_size, max_size))
            buffer = BytesIO()
            frame.save(buffer, format='JPEG', quality=88)
            payload = buffer.getvalue()
    except Exception:
        payload = _placeholder_image(max_size)

    thumb_path.write_bytes(payload)
    return StreamingResponse(BytesIO(payload), media_type='image/jpeg')


@app.post('/api/images/{image_id}/trash')
def move_to_trash(image_id: str) -> dict[str, Any]:
    config = load_config()
    recycle_bin_dir = Path(config['recycle_bin_dir']).expanduser().resolve()
    recycle_bin_dir.mkdir(parents=True, exist_ok=True)

    state = load_state()
    images = state.get('images', [])
    image = next((entry for entry in images if entry['id'] == image_id), None)
    if image is None:
        raise HTTPException(status_code=404, detail='图片不存在')

    source_path = Path(image['path'])
    if not source_path.exists():
        raise HTTPException(status_code=404, detail='源图片不存在')

    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    destination = recycle_bin_dir / f'{stamp}-{source_path.name}'
    counter = 1
    while destination.exists():
        destination = recycle_bin_dir / f'{stamp}-{counter}-{source_path.name}'
        counter += 1
    shutil.move(str(source_path), str(destination))

    state['images'] = [entry for entry in images if entry['id'] != image_id]
    state['summary'] = _summary(state['images'])
    save_state(state)

    thumb_prefix = f'{image_id}_'
    for thumb in THUMB_DIR.glob(f'{thumb_prefix}*.jpg'):
        thumb.unlink(missing_ok=True)

    return {'ok': True, 'destination': str(destination), 'state': load_state()}


@app.post('/api/trash/empty')
def empty_trash() -> dict[str, Any]:
    config = load_config()
    recycle_bin_dir = Path(config['recycle_bin_dir']).expanduser().resolve()
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

    return {'ok': True, 'removed_files': removed}


@app.post('/api/recycle-bin/open')
def open_recycle_bin() -> dict[str, Any]:
    config = load_config()
    recycle_bin_dir = Path(config['recycle_bin_dir']).expanduser().resolve()
    recycle_bin_dir.mkdir(parents=True, exist_ok=True)

    if platform.system() == 'Darwin':
        result = subprocess.run(['open', str(recycle_bin_dir)], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=result.stderr.strip() or '打开回收站失败')
        return {'ok': True, 'path': str(recycle_bin_dir)}

    raise HTTPException(status_code=501, detail='当前只实现了 macOS 打开回收站。')


@app.get('/api/health')
def health() -> JSONResponse:
    return JSONResponse({'ok': True})
