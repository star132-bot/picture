from __future__ import annotations

import threading
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import subprocess
from tkinter import filedialog, messagebox, ttk
from typing import Any, Optional

from PIL import Image, ImageFile, ImageOps, ImageTk

from .config import ensure_directories, load_config, save_config, save_state
from .library_service import LibraryService
from .scoring import MODEL_REGISTRY, NimaScorer, image_id_for_path, list_image_files

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

CARD_WIDTH = 240
THUMB_SIZE = (220, 160)
VIEWER_MAX = (1600, 1100)


class PhotoRankerDesktop:
    def __init__(self) -> None:
        ensure_directories()
        self.root = tk.Tk()
        self.root.title('Photo Ranker')
        self.root.geometry('1480x920')
        self.root.minsize(1180, 720)

        self.scorer = NimaScorer()
        self.service = LibraryService(self.scorer)
        self.config = self._normalized_config(load_config())
        self.all_images: list[dict[str, Any]] = []
        self.images: list[dict[str, Any]] = []
        self.thumbnail_refs: dict[str, ImageTk.PhotoImage] = {}
        self.viewer_image_ref: Optional[ImageTk.PhotoImage] = None
        self.viewer_window: Optional[tk.Toplevel] = None
        self.viewer_index = -1
        self.render_position = 0
        self.render_batch_size = 18
        self.is_busy = False
        self.scan_job_id = 0
        self.scan_active = False
        self.gallery_generation = 0
        self.thumb_executor = ThreadPoolExecutor(max_workers=2)

        self._build_ui()
        self._populate_config_fields()
        self.placeholder_thumb = self._build_placeholder_thumb()
        self.refresh_model_info()
        self.load_state_into_gallery()

    def _normalized_config(self, config: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(config)
        if float(normalized.get('blur_threshold', 60.0)) >= 120.0:
            normalized['blur_threshold'] = 60.0
        return normalized

    def _build_ui(self) -> None:
        self.root.configure(bg='#efe8dd')

        shell = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        shell.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(shell, padding=16)
        right = ttk.Frame(shell, padding=16)
        shell.add(left, weight=0)
        shell.add(right, weight=1)

        title = ttk.Label(left, text='Photo Ranker', font=('Helvetica Neue', 24, 'bold'))
        title.pack(anchor='w')
        ttk.Label(left, text='本地桌面筛片工具，不再依赖浏览器。').pack(anchor='w', pady=(4, 14))

        settings = ttk.LabelFrame(left, text='扫描设置', padding=12)
        settings.pack(fill=tk.X)

        self.image_dir_var = tk.StringVar()
        self.recycle_dir_var = tk.StringVar()
        self.model_var = tk.StringVar()
        self.blur_var = tk.DoubleVar()
        self.thumb_var = tk.IntVar()
        self.batch_var = tk.IntVar()
        self.status_var = tk.StringVar(value='准备就绪')
        self.summary_var = tk.StringVar(value='等待扫描')

        self._path_field(settings, '图片目录', self.image_dir_var, self.pick_image_dir)
        self._path_field(settings, '回收站目录', self.recycle_dir_var, self.pick_recycle_dir)

        row1 = ttk.Frame(settings)
        row1.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(row1, text='模型').grid(row=0, column=0, sticky='w')
        self.model_box = ttk.Combobox(row1, textvariable=self.model_var, state='readonly', values=['mobilenet', 'inception'], width=18)
        self.model_box.grid(row=1, column=0, sticky='ew', padx=(0, 8))
        self.model_box.bind('<<ComboboxSelected>>', lambda _event: self.refresh_model_info())
        ttk.Label(row1, text='模糊阈值').grid(row=0, column=1, sticky='w')
        ttk.Spinbox(row1, from_=0, to=300, increment=5, textvariable=self.blur_var, width=10).grid(row=1, column=1, sticky='ew')
        row1.columnconfigure(0, weight=1)
        row1.columnconfigure(1, weight=1)

        row2 = ttk.Frame(settings)
        row2.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(row2, text='缩略图尺寸').grid(row=0, column=0, sticky='w')
        ttk.Spinbox(row2, from_=240, to=1200, increment=20, textvariable=self.thumb_var, width=10).grid(row=1, column=0, sticky='ew', padx=(0, 8))
        ttk.Label(row2, text='批大小').grid(row=0, column=1, sticky='w')
        ttk.Spinbox(row2, from_=1, to=128, increment=1, textvariable=self.batch_var, width=10).grid(row=1, column=1, sticky='ew')
        row2.columnconfigure(0, weight=1)
        row2.columnconfigure(1, weight=1)

        ttk.Label(settings, text='建议模糊阈值先用 30-80。数值越高，隐藏越多。').pack(anchor='w', pady=(10, 0))

        buttons = ttk.Frame(settings)
        buttons.pack(fill=tk.X, pady=(12, 0))
        self.save_btn = ttk.Button(buttons, text='保存设置', command=self.save_current_config)
        self.save_btn.pack(side=tk.LEFT)
        self.scan_btn = ttk.Button(buttons, text='开始扫描', command=self.run_scan)
        self.scan_btn.pack(side=tk.LEFT, padx=8)

        models_frame = ttk.LabelFrame(left, text='模型状态', padding=12)
        models_frame.pack(fill=tk.X, pady=(14, 0))
        self.model_info_label = ttk.Label(models_frame, text='', justify=tk.LEFT)
        self.model_info_label.pack(anchor='w')
        dl_row = ttk.Frame(models_frame)
        dl_row.pack(fill=tk.X, pady=(10, 0))
        self.download_mobile_btn = ttk.Button(dl_row, text='下载 MobileNet', command=lambda: self.download_model('mobilenet'))
        self.download_mobile_btn.pack(side=tk.LEFT)
        self.download_inception_btn = ttk.Button(dl_row, text='下载 Inception', command=lambda: self.download_model('inception'))
        self.download_inception_btn.pack(side=tk.LEFT, padx=8)

        actions_frame = ttk.LabelFrame(left, text='快捷操作', padding=12)
        actions_frame.pack(fill=tk.X, pady=(14, 0))
        ttk.Label(actions_frame, text='空格 删除当前查看图到回收站\n← / → 上一张 / 下一张\nEsc 退出全屏查看').pack(anchor='w')
        ttk.Button(actions_frame, text='打开回收站', command=self.open_recycle_bin).pack(anchor='w', pady=(10, 0))
        ttk.Button(actions_frame, text='清空回收站', command=self.empty_trash).pack(anchor='w', pady=(10, 0))

        meta_frame = ttk.LabelFrame(left, text='运行状态', padding=12)
        meta_frame.pack(fill=tk.BOTH, expand=True, pady=(14, 0))
        ttk.Label(meta_frame, textvariable=self.status_var, wraplength=300, justify=tk.LEFT).pack(anchor='w')
        ttk.Label(meta_frame, textvariable=self.summary_var, wraplength=300, justify=tk.LEFT).pack(anchor='w', pady=(10, 0))

        header = ttk.Frame(right)
        header.pack(fill=tk.X)
        ttk.Label(header, text='评分结果', font=('Helvetica Neue', 20, 'bold')).pack(side=tk.LEFT)
        header_tools = ttk.Frame(header)
        header_tools.pack(side=tk.RIGHT)
        ttk.Label(header_tools, text='视图').pack(side=tk.LEFT)
        self.view_mode_var = tk.StringVar(value='正常视图')
        self.view_mode_box = ttk.Combobox(
            header_tools,
            textvariable=self.view_mode_var,
            state='readonly',
            values=['正常视图', '模糊视图', '全部图片'],
            width=10,
        )
        self.view_mode_box.pack(side=tk.LEFT, padx=(6, 10))
        self.view_mode_box.bind('<<ComboboxSelected>>', lambda _event: self.refresh_gallery_view())

        ttk.Label(header_tools, text='排序').pack(side=tk.LEFT)
        self.sort_mode_var = tk.StringVar(value='按排名')
        self.sort_mode_box = ttk.Combobox(
            header_tools,
            textvariable=self.sort_mode_var,
            state='readonly',
            values=['按排名', '按文件名'],
            width=10,
        )
        self.sort_mode_box.pack(side=tk.LEFT, padx=(6, 10))
        self.sort_mode_box.bind('<<ComboboxSelected>>', lambda _event: self.refresh_gallery_view())

        self.gallery_meta = ttk.Label(header_tools, text='')
        self.gallery_meta.pack(side=tk.LEFT)

        gallery_shell = ttk.Frame(right)
        gallery_shell.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        self.gallery_canvas = tk.Canvas(gallery_shell, bg='#f6f1ea', highlightthickness=0)
        self.gallery_scroll = ttk.Scrollbar(gallery_shell, orient=tk.VERTICAL, command=self.gallery_canvas.yview)
        self.gallery_canvas.configure(yscrollcommand=self.gallery_scroll.set)
        self.gallery_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.gallery_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.gallery_frame = ttk.Frame(self.gallery_canvas)
        self.gallery_window = self.gallery_canvas.create_window((0, 0), window=self.gallery_frame, anchor='nw')
        self.gallery_frame.bind('<Configure>', self._sync_canvas_region)
        self.gallery_canvas.bind('<Configure>', self._resize_gallery_window)
        self.gallery_canvas.bind_all('<MouseWheel>', self._on_mousewheel)

    def _path_field(self, parent: ttk.Frame, label: str, variable: tk.StringVar, callback) -> None:
        ttk.Label(parent, text=label).pack(anchor='w', pady=(8, 0))
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=(4, 0))
        ttk.Entry(row, textvariable=variable).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text='选择文件夹', command=callback).pack(side=tk.LEFT, padx=(8, 0))

    def _sync_canvas_region(self, _event=None) -> None:
        self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox('all'))

    def _resize_gallery_window(self, event) -> None:
        self.gallery_canvas.itemconfigure(self.gallery_window, width=event.width)

    def _on_mousewheel(self, event) -> None:
        self.gallery_canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')

    def _build_placeholder_thumb(self) -> ImageTk.PhotoImage:
        canvas = Image.new('RGB', THUMB_SIZE, '#ddd4c7')
        return ImageTk.PhotoImage(canvas)

    def _populate_config_fields(self) -> None:
        self.image_dir_var.set(self.config['image_dir'])
        self.recycle_dir_var.set(self.config['recycle_bin_dir'])
        self.model_var.set(self.config['model_name'])
        self.blur_var.set(float(self.config['blur_threshold']))
        self.thumb_var.set(int(self.config['thumbnail_size']))
        self.batch_var.set(int(self.config['scan_batch_size']))

    def gather_config(self) -> dict[str, Any]:
        return {
            'image_dir': self.image_dir_var.get().strip(),
            'recycle_bin_dir': self.recycle_dir_var.get().strip(),
            'model_name': self.model_var.get().strip() or 'mobilenet',
            'blur_threshold': float(self.blur_var.get() or 60.0),
            'thumbnail_size': int(self.thumb_var.get() or 520),
            'scan_batch_size': int(self.batch_var.get() or 24),
        }

    def save_current_config(self) -> dict[str, Any]:
        self.config = self._normalized_config(save_config(self.gather_config()))
        self.status_var.set('设置已保存。')
        return self.config

    def pick_image_dir(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.image_dir_var.get() or str(Path.home()), title='选择图片目录')
        if chosen:
            self.image_dir_var.set(chosen)
            self.save_current_config()

    def pick_recycle_dir(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.recycle_dir_var.get() or str(Path.home()), title='选择回收站目录')
        if chosen:
            self.recycle_dir_var.set(chosen)
            self.save_current_config()

    def set_busy(self, busy: bool, message: Optional[str] = None) -> None:
        self.is_busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        for widget in [self.save_btn, self.scan_btn, self.download_mobile_btn, self.download_inception_btn]:
            widget.configure(state=state)
        self.model_box.configure(state=tk.DISABLED if busy else 'readonly')
        self.view_mode_box.configure(state=tk.DISABLED if busy else 'readonly')
        self.sort_mode_box.configure(state=tk.DISABLED if busy else 'readonly')
        if message:
            self.status_var.set(message)

    def run_background(self, message: str, worker, on_success) -> None:
        if self.is_busy:
            return
        self.set_busy(True, message)

        def task() -> None:
            try:
                result = worker()
            except Exception as exc:
                self.root.after(0, lambda: self._background_error(str(exc)))
                return
            self.root.after(0, lambda: self._background_success(on_success, result))

        threading.Thread(target=task, daemon=True).start()

    def _background_error(self, error: str) -> None:
        self.scan_active = False
        self.set_busy(False, f'操作失败: {error}')
        messagebox.showerror('Photo Ranker', error)

    def _background_success(self, callback, result) -> None:
        self.set_busy(False)
        callback(result)

    def refresh_model_info(self) -> None:
        models = self.scorer.list_models()
        runtime = self.scorer.runtime_info()
        selected_model = self.model_var.get().strip() or self.config.get('model_name', 'mobilenet')
        lines = [
            f"当前模型: {MODEL_REGISTRY[selected_model]['label']}",
            f"MPS 构建: {'是' if runtime['mps_built'] else '否'}",
            f"MPS 可用: {'是' if runtime['mps_available'] else '否'}",
        ]
        for model in models:
            status = '已下载' if model['weight_exists'] else '未下载'
            size = f" {model['weight_size_mb']}MB" if model['weight_exists'] else ''
            lines.append(f"{model['label']}: {status}{size}")
        self.model_info_label.configure(text='\n'.join(lines))

    def download_model(self, model_name: str) -> None:
        self.run_background(
            f'正在下载 {MODEL_REGISTRY[model_name]["label"]} 权重……',
            lambda: self.scorer.download_weights(model_name),
            lambda result: self._download_finished(result),
        )

    def _download_finished(self, result: dict[str, Any]) -> None:
        self.refresh_model_info()
        self.status_var.set(f"{result['model_name']} 权重下载完成：{result['downloaded_mb']} MB")

    def run_scan(self) -> None:
        config = self.save_current_config()
        if self.is_busy:
            return
        self.scan_job_id += 1
        job_id = self.scan_job_id
        self.scan_active = True
        self.all_images = []
        self.images = []
        self.close_viewer()
        self.view_mode_var.set('全部图片')
        self.render_gallery_lazy()
        self.set_busy(True, '正在扫描并评分，请稍等……')
        self.summary_var.set('正在建立预览列表…')

        def task() -> None:
            try:
                result = self._scan_library_progressive(config, job_id)
            except Exception as exc:
                self.root.after(0, lambda: self._background_error(str(exc)))
                return
            self.root.after(0, lambda: self._scan_finished(job_id, result))

        threading.Thread(target=task, daemon=True).start()

    def _scan_library_progressive(self, config: dict[str, Any], job_id: int) -> dict[str, Any]:
        image_dir = Path(config['image_dir']).expanduser().resolve()
        recycle_bin_dir = Path(config['recycle_bin_dir']).expanduser().resolve()
        files = list_image_files(image_dir, recycle_bin_dir)
        total = len(files)
        batch_size = max(1, int(config['scan_batch_size']))
        blur_threshold = float(config['blur_threshold'])
        bundle, warnings, source = self.scorer.prepare_scoring(config['model_name'])

        images = self._placeholder_images(files)
        self.root.after(0, lambda initial=[dict(item) for item in images], scan_id=job_id, count=total: self._seed_scan_placeholders(scan_id, initial, count))
        image_index = {image['id']: image for image in images}
        for start in range(0, total, batch_size):
            batch_paths = files[start:start + batch_size]
            metadata = self.scorer.inspect_images(batch_paths, blur_threshold)
            scored_batch = self.scorer.score_metadata(metadata, batch_size, bundle, source)
            for scored_image in scored_batch:
                image_index[scored_image['id']].update(scored_image)
            processed = min(start + len(batch_paths), total)
            self.root.after(
                0,
                lambda batch=[dict(item) for item in images], done=processed, count=total, scan_id=job_id, scan_source=source: self._scan_progress_update(
                    scan_id, batch, done, count, scan_source
                ),
            )

        ranked_images = self.scorer.rank_images([dict(item) for item in images])
        state = {
            'images': ranked_images,
            'summary': self.service.summary(ranked_images),
            'scanned_at': datetime.now().isoformat(timespec='seconds'),
            'scan_details': {
                'warnings': warnings,
                'score_source': source,
            },
        }
        save_state(state)
        return state

    def _placeholder_images(self, files: list[Path]) -> list[dict[str, Any]]:
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

    def _seed_scan_placeholders(self, job_id: int, images: list[dict[str, Any]], total: int) -> None:
        if job_id != self.scan_job_id:
            return
        self.all_images = images
        self.status_var.set(f'已载入预览 0/{total}')
        self.summary_var.set(f'已显示 {total} 张图片预览，正在后台评分…')
        self.refresh_gallery_view()

    def _scan_progress_update(self, job_id: int, images: list[dict[str, Any]], processed: int, total: int, source: str) -> None:
        if job_id != self.scan_job_id:
            return
        self.all_images = images
        summary = self.service.summary(images)
        self.status_var.set(f'正在扫描并评分… {processed}/{total}')
        self.refresh_gallery_view()
        self.summary_var.set(
            f"预览已显示 {total} 张 | 已评分 {processed}/{total} | "
            f"已识别模糊 {summary.get('hidden_blurry', 0)} | 评分来源 {source}"
        )

    def _scan_finished(self, job_id: int, state: dict[str, Any]) -> None:
        if job_id != self.scan_job_id:
            return
        self.scan_active = False
        self.set_busy(False)
        self.all_images = state.get('images', [])
        summary = state.get('summary', {})
        if summary.get('visible', 0) == 0 and summary.get('hidden_blurry', 0) > 0:
            self.view_mode_var.set('模糊视图')
            self.status_var.set('扫描完成。当前图片都被模糊规则隐藏，已自动切到模糊视图。')
        else:
            self.view_mode_var.set('正常视图')
            self.status_var.set('扫描完成。')
        self.sort_mode_var.set('按排名')
        self.refresh_gallery_view()

    def load_state_into_gallery(self) -> None:
        payload = self.service.load_images(include_hidden=True)
        self.all_images = payload['images']
        self.refresh_gallery_view(payload)

    def refresh_gallery_view(self, payload: Optional[dict[str, Any]] = None) -> None:
        if payload is None:
            payload = {
                'images': self.all_images,
                'summary': self.service.summary(self.all_images),
                'view_count': len(self.all_images),
                'scan_details': {},
            }
        all_images = self.all_images or payload.get('images', [])
        self.images = self._sorted_images(self._filtered_images(all_images))
        summary = payload['summary']
        sort_label = '扫描顺序' if self.scan_active else self.sort_mode_var.get()
        self.gallery_meta.configure(
            text=f"总数 {summary.get('total', 0)} · 当前视图 {len(self.images)} · {self.view_mode_var.get()} · {sort_label}"
        )
        self.summary_var.set(
            f"显示 {summary.get('visible', 0)} | 隐藏模糊 {summary.get('hidden_blurry', 0)} | "
            f"最高分 {summary.get('top_score') if summary.get('top_score') is not None else '-'} | "
            f"评分来源 {summary.get('score_source') or '-'}"
        )
        self.render_gallery_lazy()

    def _filtered_images(self, images: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.scan_active:
            return list(images)
        mode = self.view_mode_var.get()
        if mode == '模糊视图':
            return [image for image in images if image.get('is_blurry')]
        if mode == '全部图片':
            return list(images)
        return [image for image in images if not image.get('is_blurry')]

    def _sorted_images(self, images: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.scan_active:
            return sorted(images, key=lambda image: image.get('scan_index', image.get('rank', 10 ** 9)))
        if self.sort_mode_var.get() == '按文件名':
            return sorted(images, key=lambda image: image['filename'].casefold())
        return sorted(images, key=lambda image: (image.get('rank', 10 ** 9), image['filename'].casefold()))

    def render_gallery_lazy(self) -> None:
        self.gallery_generation += 1
        for child in self.gallery_frame.winfo_children():
            child.destroy()
        self.thumbnail_refs.clear()
        self.render_position = 0
        if not self.images:
            msg = '当前没有图片。请先选择目录并扫描。'
            if self.scan_active:
                msg = '正在建立预览列表…'
            summary = self.service.summary(self.all_images)
            if summary.get('hidden_blurry', 0) > 0 and self.view_mode_var.get() == '正常视图':
                msg = f"当前正常视图没有图片，已有 {summary['hidden_blurry']} 张在模糊视图里。请切到“模糊视图”或“全部图片”。"
            ttk.Label(self.gallery_frame, text=msg, padding=16).grid(row=0, column=0, sticky='w')
            return
        self._render_next_batch()

    def _render_next_batch(self) -> None:
        end = min(self.render_position + self.render_batch_size, len(self.images))
        for index in range(self.render_position, end):
            self._render_card(self.images[index], index, self.gallery_generation)
        self.render_position = end
        if self.render_position < len(self.images):
            self.root.after(10, self._render_next_batch)

    def _render_card(self, image: dict[str, Any], index: int, generation: int) -> None:
        columns = max(1, self.gallery_canvas.winfo_width() // (CARD_WIDTH + 12))
        row = index // columns
        column = index % columns
        card = ttk.Frame(self.gallery_frame, relief=tk.RIDGE, padding=10)
        card.grid(row=row, column=column, padx=6, pady=6, sticky='nsew')
        self.gallery_frame.columnconfigure(column, weight=1)

        thumb_label = tk.Label(card, cursor='hand2', bg='#ddd4c7')
        thumb_label.pack(fill=tk.BOTH)
        thumb_label.bind('<Button-1>', lambda _e, idx=index: self.open_viewer(idx))
        thumb_label.configure(image=self.placeholder_thumb)
        self._queue_thumbnail_load(Path(image['path']), image['id'], thumb_label, generation)

        rank_text = f"#{image.get('rank', image.get('scan_index', index + 1))}"
        score_text = '扫描中' if image.get('score') is None else f"{image['score']:.2f}"
        ttk.Label(card, text=f"{rank_text} · {score_text}", font=('Helvetica Neue', 13, 'bold')).pack(anchor='w', pady=(8, 0))
        ttk.Label(card, text=image['filename'], wraplength=CARD_WIDTH - 20).pack(anchor='w', pady=(4, 0))
        tags = []
        if image.get('score') is None:
            tags.append('等待评分')
        elif image.get('is_blurry'):
            tags.append('模糊隐藏')
        if image.get('blur_score') is not None:
            tags.append(f"清晰度 {image['blur_score']:.1f}")
        if image.get('width') and image.get('height'):
            tags.append(f"{image['width']}×{image['height']}")
        ttk.Label(card, text=' · '.join(tags), wraplength=CARD_WIDTH - 20).pack(anchor='w', pady=(4, 0))

        action_row = ttk.Frame(card)
        action_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(action_row, text='查看', command=lambda idx=index: self.open_viewer(idx)).pack(side=tk.LEFT)
        ttk.Button(action_row, text='移到回收站', command=lambda img_id=image['id']: self.delete_image(img_id)).pack(side=tk.RIGHT)

    def _load_thumbnail_canvas(self, path: Path) -> Image.Image:
        canvas = Image.new('RGB', THUMB_SIZE, '#ddd4c7')
        try:
            with Image.open(path) as img:
                frame = ImageOps.exif_transpose(img.convert('RGB'))
                frame.thumbnail(THUMB_SIZE)
        except Exception:
            frame = None

        if frame is not None:
            offset = ((THUMB_SIZE[0] - frame.width) // 2, (THUMB_SIZE[1] - frame.height) // 2)
            canvas.paste(frame, offset)
        return canvas

    def _queue_thumbnail_load(self, path: Path, image_id: str, label: tk.Label, generation: int) -> None:
        def task() -> None:
            canvas = self._load_thumbnail_canvas(path)
            self.root.after(0, lambda: self._apply_thumbnail(canvas, image_id, label, generation))

        self.thumb_executor.submit(task)

    def _apply_thumbnail(self, canvas: Image.Image, image_id: str, label: tk.Label, generation: int) -> None:
        if generation != self.gallery_generation or not label.winfo_exists():
            return
        photo = ImageTk.PhotoImage(canvas)
        label.configure(image=photo)
        self.thumbnail_refs[image_id] = photo

    def delete_image(self, image_id: str) -> None:
        if self.is_busy:
            return
        if not messagebox.askyesno('Photo Ranker', '确认把这张图片移动到回收站吗？'):
            return
        config = self.save_current_config()
        self.run_background('正在移动图片到回收站……', lambda: self.service.move_to_trash(image_id, config), self._delete_finished)

    def _delete_finished(self, destination: str) -> None:
        self.status_var.set(f'已移动到回收站：{destination}')
        self.load_state_into_gallery()
        if self.viewer_window and self.viewer_window.winfo_exists():
            self.close_viewer()

    def empty_trash(self) -> None:
        if self.is_busy:
            return
        if not messagebox.askyesno('Photo Ranker', '确认清空回收站吗？该操作不可恢复。'):
            return
        config = self.save_current_config()
        self.run_background('正在清空回收站……', lambda: self.service.empty_trash(config), self._trash_finished)

    def _trash_finished(self, removed: int) -> None:
        self.status_var.set(f'回收站已清空，删除 {removed} 个文件。')

    def open_recycle_bin(self) -> None:
        recycle_bin_dir = Path(self.save_current_config()['recycle_bin_dir']).expanduser().resolve()
        recycle_bin_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(['open', str(recycle_bin_dir)], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            messagebox.showerror('Photo Ranker', result.stderr.strip() or '打开回收站失败')
            return
        self.status_var.set(f'已打开回收站：{recycle_bin_dir}')

    def open_viewer(self, index: int) -> None:
        if index < 0 or index >= len(self.images):
            return
        self.viewer_index = index
        if self.viewer_window is None or not self.viewer_window.winfo_exists():
            self.viewer_window = tk.Toplevel(self.root)
            self.viewer_window.configure(bg='black')
            self.viewer_window.attributes('-fullscreen', True)
            self.viewer_window.bind('<Escape>', lambda _e: self.close_viewer())
            self.viewer_window.bind('<Left>', lambda _e: self.shift_viewer(-1))
            self.viewer_window.bind('<Right>', lambda _e: self.shift_viewer(1))
            self.viewer_window.bind('<space>', lambda _e: self.delete_current_viewer_image())

            top = tk.Frame(self.viewer_window, bg='black')
            top.pack(fill=tk.BOTH, expand=True)
            self.viewer_prev_btn = ttk.Button(top, text='上一张', command=lambda: self.shift_viewer(-1))
            self.viewer_prev_btn.place(relx=0.02, rely=0.5, anchor='w')
            self.viewer_next_btn = ttk.Button(top, text='下一张', command=lambda: self.shift_viewer(1))
            self.viewer_next_btn.place(relx=0.98, rely=0.5, anchor='e')
            self.viewer_close_btn = ttk.Button(top, text='关闭', command=self.close_viewer)
            self.viewer_close_btn.place(relx=0.98, rely=0.04, anchor='ne')
            self.viewer_label = tk.Label(top, bg='black')
            self.viewer_label.pack(fill=tk.BOTH, expand=True)
            self.viewer_caption = tk.Label(top, bg='black', fg='white', font=('Helvetica Neue', 14))
            self.viewer_caption.pack(side=tk.BOTTOM, pady=18)
        self._show_viewer_image()

    def _show_viewer_image(self) -> None:
        image = self.images[self.viewer_index]
        path = Path(image['path'])
        with Image.open(path) as img:
            frame = ImageOps.exif_transpose(img.convert('RGB'))
            frame.thumbnail(VIEWER_MAX)
        self.viewer_image_ref = ImageTk.PhotoImage(frame)
        self.viewer_label.configure(image=self.viewer_image_ref)
        score_text = '扫描中' if image.get('score') is None else f"{image['score']:.2f} 分"
        blur_text = '清晰度待计算' if image.get('blur_score') is None else f"清晰度 {image['blur_score']:.1f}"
        self.viewer_caption.configure(
            text=f"#{image.get('rank', image.get('scan_index', self.viewer_index + 1))} · {image['filename']} · {score_text} · {blur_text}"
        )

    def shift_viewer(self, step: int) -> None:
        if not self.images:
            return
        self.viewer_index = (self.viewer_index + step + len(self.images)) % len(self.images)
        self._show_viewer_image()

    def delete_current_viewer_image(self) -> None:
        if 0 <= self.viewer_index < len(self.images):
            image_id = self.images[self.viewer_index]['id']
            self.delete_image(image_id)

    def close_viewer(self) -> None:
        if self.viewer_window and self.viewer_window.winfo_exists():
            self.viewer_window.destroy()
        self.viewer_window = None
        self.viewer_index = -1
        self.viewer_image_ref = None

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = PhotoRankerDesktop()
    app.run()


if __name__ == '__main__':
    main()
