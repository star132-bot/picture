# Photo Ranker

Photo Ranker is a local-first image review tool for quickly sorting large photo folders by aesthetic score and blur quality.

It includes:

- A browser-based workflow that starts fast and works well for large batches
- An optional Tk desktop UI on macOS when the local Python/Tk environment supports it
- NIMA-based scoring with `MobileNet` and `Inception ResNet V2`
- Blur detection, hidden-blurry view, and quick move-to-trash actions
- Progressive scanning: images appear first, scores are filled in while scanning

## Features

- Rank images by score from high to low
- Switch between normal view, blurry view, and all images
- Scan large folders with progressive preview loading
- Open, empty, and use a configurable recycle bin folder
- Fullscreen preview with keyboard navigation
- Download model weights from inside the app when needed

## Screens and Modes

### Web UI

Recommended for most users.

- Fast startup
- Progressive scan previews
- Sorting after scan completion
- Recycle bin actions from the browser UI

### Desktop UI

Available through `app.desktop` when `tkinter` works correctly on the host machine.

## Requirements

- Python `3.9+`
- macOS recommended for the bundled launchers
- Dependencies from `requirements.txt`

## Installation

```bash
git clone https://github.com/star132-bot/picture.git
cd picture
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Start

### Option 1: macOS one-click launcher

Double-click:

- `disk/一键启动 Photo Ranker.command`

The launcher:

- prefers `.venv/bin/python` when available
- falls back to `python3`
- tries the desktop app first
- falls back to the web app automatically if Tk is unavailable

### Option 2: Web UI

```bash
cd /path/to/picture
source .venv/bin/activate
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8123
```

Then open:

- `http://127.0.0.1:8123`

### Option 3: Desktop UI

```bash
cd /path/to/picture
source .venv/bin/activate
python3 -m app.desktop
```

## First Run

1. Choose an image folder.
2. Choose a recycle bin folder.
3. Select a model:
   - `mobilenet`: faster
   - `inception`: more expensive, often better for final review
4. Start scanning.
5. Wait for scores to complete, then sort and review.

## Models

This repository does **not** ship model weights by default.

Weights are downloaded at runtime into `models/`:

- `mobilenet_weights.h5`
- `inception_resnet_weights.h5`

That keeps the repository lightweight and GitHub-friendly.

## Project Structure

- `app/main.py`: FastAPI web app
- `app/desktop.py`: Tk desktop app
- `app/scoring.py`: image scoring, blur analysis, ranking
- `app/library_service.py`: scan state and recycle bin operations
- `app/config.py`: runtime config and local state
- `app/static/`: browser UI assets
- `app/templates/`: browser UI templates
- `disk/`: launchers
- `scripts/`: helper start script
- `sample_images/`: small demo images

## Notes

- Runtime-generated files are stored under `data/` and are intentionally ignored by Git.
- Downloaded model files under `models/` are also ignored by Git.
- The recycle bin opener currently uses macOS `open`.

## License

MIT
