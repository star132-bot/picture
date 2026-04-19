const state = {
  images: [],
  config: null,
  models: [],
  runtime: null,
  currentIndex: -1,
  viewMode: 'visible',
  sortMode: 'rank',
  scanActive: false,
  scanId: null,
};

const els = {
  imageDir: document.getElementById('imageDir'),
  recycleBinDir: document.getElementById('recycleBinDir'),
  modelName: document.getElementById('modelName'),
  blurThreshold: document.getElementById('blurThreshold'),
  thumbSize: document.getElementById('thumbSize'),
  scanBatchSize: document.getElementById('scanBatchSize'),
  modelInfo: document.getElementById('modelInfo'),
  statusMeta: document.getElementById('statusMeta'),
  feedback: document.getElementById('feedback'),
  pickImageDirBtn: document.getElementById('pickImageDirBtn'),
  pickRecycleBinDirBtn: document.getElementById('pickRecycleBinDirBtn'),
  gallery: document.getElementById('gallery'),
  galleryTitle: document.getElementById('galleryTitle'),
  summaryBar: document.getElementById('summaryBar'),
  viewMode: document.getElementById('viewMode'),
  sortMode: document.getElementById('sortMode'),
  saveConfigBtn: document.getElementById('saveConfigBtn'),
  scanBtn: document.getElementById('scanBtn'),
  downloadBtn: document.getElementById('downloadBtn'),
  openRecycleBinBtn: document.getElementById('openRecycleBinBtn'),
  emptyTrashBtn: document.getElementById('emptyTrashBtn'),
  viewer: document.getElementById('viewer'),
  viewerImage: document.getElementById('viewerImage'),
  viewerCaption: document.getElementById('viewerCaption'),
  viewerClose: document.getElementById('viewerClose'),
  viewerPrev: document.getElementById('viewerPrev'),
  viewerNext: document.getElementById('viewerNext'),
};

async function request(url, options = {}) {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `HTTP ${response.status}`);
  }
  return response.json();
}

function configPayload() {
  return {
    image_dir: els.imageDir.value.trim(),
    recycle_bin_dir: els.recycleBinDir.value.trim(),
    model_name: els.modelName.value,
    blur_threshold: Number(els.blurThreshold.value || 120),
    thumbnail_size: Number(els.thumbSize.value || 520),
    scan_batch_size: Number(els.scanBatchSize.value || 24),
  };
}

function setFeedback(message, isError = false) {
  els.feedback.textContent = message || '';
  els.feedback.style.color = isError ? 'var(--danger)' : 'var(--accent)';
}

function syncForm(config) {
  state.config = config;
  els.imageDir.value = config.image_dir || '';
  els.recycleBinDir.value = config.recycle_bin_dir || '';
  els.modelName.value = config.model_name || 'mobilenet';
  els.blurThreshold.value = config.blur_threshold ?? 120;
  els.thumbSize.value = config.thumbnail_size ?? 520;
  els.scanBatchSize.value = config.scan_batch_size ?? 24;
}

function renderModels() {
  const current = els.modelName.value;
  els.modelInfo.innerHTML = state.models.map(model => {
    const active = model.name === current ? '当前选择' : '';
    const ready = model.weight_exists ? `已下载 ${model.weight_size_mb} MB` : '未下载';
    return `
      <div>
        <strong>${model.label}</strong> ${active ? `· ${active}` : ''}<br />
        <span>${model.speed_note}</span><br />
        <span>权重: ${ready}</span>
      </div>
    `;
  }).join('');
}

function renderStatus(meta) {
  const runtime = state.runtime || {};
  const modelMode = meta?.scan_details?.score_source || '未扫描';
  const scannedAt = meta?.scanned_at || '未扫描';
  const warnings = meta?.scan_details?.warnings || [];
  const progress = meta?.total ? `${meta.processed || 0}/${meta.total}` : null;
  els.statusMeta.innerHTML = `
    <p>MPS 构建: ${runtime.mps_built ? '是' : '否'}</p>
    <p>MPS 可用: ${runtime.mps_available ? '是' : '否'}</p>
    <p>TensorFlow GPU: ${(runtime.tensorflow_gpus || []).length}</p>
    ${progress ? `<p>扫描进度: ${progress}</p>` : ''}
    <p>评分来源: ${modelMode}</p>
    <p>最近扫描: ${scannedAt}</p>
    ${warnings.length ? `<p>提示: ${warnings.join(' / ')}</p>` : ''}
  `;
}

function renderSummary(summary = {}, scannedAt = '', viewCount = 0) {
  const items = [
    `总数 ${summary.total || 0}`,
    `当前视图 ${viewCount || 0}`,
    `显示 ${summary.visible || 0}`,
    `隐藏模糊 ${summary.hidden_blurry || 0}`,
    `最高分 ${summary.top_score ? summary.top_score.toFixed(2) : '-'}`,
  ];
  if (scannedAt) {
    items.push(scannedAt.replace('T', ' '));
  }
  els.summaryBar.innerHTML = items.map(item => `<div class="summary-pill">${item}</div>`).join('');
}

function imageCard(image, index) {
  const blurTag = image.is_blurry ? `<span class="blur-tag">模糊隐藏</span>` : '';
  const detail = image.error
    ? `读取失败 · ${image.error}`
    : image.score == null
      ? '等待评分'
      : `${blurTag} 清晰度 ${image.blur_score.toFixed(1)} · ${image.width}×${image.height}`;
  const scoreText = image.score == null ? '扫描中' : image.score.toFixed(2);
  const rankText = image.rank || image.scan_index || index + 1;
  return `
    <article class="card" data-index="${index}">
      <div class="card-image" data-open="${index}">
        <div class="score-badge">#${rankText} · ${scoreText}</div>
        <img loading="lazy" src="/api/media/${image.id}?variant=thumb&max_size=${state.config.thumbnail_size}" alt="${image.filename}" />
      </div>
      <div class="card-body">
        <div class="card-name">${image.filename}</div>
        <p class="muted">${detail}</p>
        <div class="card-meta">
          <span class="muted">${image.score_source}</span>
          <button data-trash="${image.id}">删除到回收站</button>
        </div>
      </div>
    </article>
  `;
}

function filteredAndSortedImages(images) {
  if (state.scanActive) {
    return [...images].sort((left, right) => (left.scan_index || left.rank || Number.MAX_SAFE_INTEGER) - (right.scan_index || right.rank || Number.MAX_SAFE_INTEGER));
  }
  let result = images;
  if (state.viewMode === 'visible') {
    result = images.filter(image => !image.is_blurry);
  } else if (state.viewMode === 'blurry') {
    result = images.filter(image => image.is_blurry);
  }

  if (state.sortMode === 'filename') {
    result = [...result].sort((left, right) => left.filename.localeCompare(right.filename, 'zh-CN'));
  } else {
    result = [...result].sort((left, right) => (left.rank || Number.MAX_SAFE_INTEGER) - (right.rank || Number.MAX_SAFE_INTEGER));
  }
  return result;
}

function renderGallery(images, meta = {}) {
  state.images = images;
  const viewTitles = {
    visible: '正常视图',
    blurry: '模糊视图',
    all: '全部图片',
  };
  const modeText = state.scanActive ? '扫描顺序' : (state.sortMode === 'rank' ? '按排名' : '按文件名');
  const titleText = state.scanActive ? '扫描中预览' : viewTitles[state.viewMode];
  els.galleryTitle.textContent = `${titleText} · ${modeText}`;
  renderSummary(meta.summary, meta.scanned_at, meta.view_count || images.length);
  if (!images.length) {
    els.gallery.innerHTML = '<div class="panel">当前没有可显示的图片。先设置目录并执行扫描。</div>';
    return;
  }
  els.gallery.innerHTML = images.map((image, index) => imageCard(image, index)).join('');
}

function setScanControlsDisabled(disabled) {
  els.viewMode.disabled = disabled;
  els.sortMode.disabled = disabled;
}

function autoSwitchHiddenView(meta = {}) {
  const summary = meta.summary || {};
  if (state.viewMode !== 'visible' || summary.visible || !summary.hidden_blurry) {
    return false;
  }
  state.viewMode = 'blurry';
  els.viewMode.value = state.viewMode;
  return true;
}

async function refreshStatus() {
  const payload = await request('/api/status');
  syncForm(payload.config);
  state.models = payload.models;
  state.runtime = payload.runtime;
  renderModels();
  renderStatus(payload.state);
  autoSwitchHiddenView(payload.state);
  const images = filteredAndSortedImages(payload.state.images || []);
  renderGallery(images, { ...payload.state, view_count: images.length });
}

async function refreshImages() {
  const payload = await request('/api/images?include_hidden=true');
  const images = filteredAndSortedImages(payload.images);
  renderGallery(images, { ...payload, view_count: images.length });
  renderStatus({ ...payload, scan_details: payload.scan_details });
}

async function persistConfig() {
  const response = await request('/api/config', {
    method: 'POST',
    body: JSON.stringify(configPayload()),
  });
  syncForm(response.config);
  renderModels();
  return response.config;
}

async function saveConfig() {
  await persistConfig();
  setFeedback('设置已保存。');
}

async function pickFolder(targetInput) {
  const currentPath = targetInput.value.trim();
  setFeedback('正在打开文件夹选择器……');
  const query = currentPath ? `?initial_path=${encodeURIComponent(currentPath)}` : '';
  const payload = await request(`/api/dialog/folder${query}`);
  if (!payload.path) {
    setFeedback('已取消文件夹选择。');
    return;
  }
  targetInput.value = payload.path;
  await persistConfig();
  setFeedback(`已选择并保存文件夹：${payload.path}`);
}

async function runScan() {
  await persistConfig();
  state.scanActive = true;
  state.scanId = null;
  state.viewMode = 'all';
  els.viewMode.value = 'all';
  setScanControlsDisabled(true);
  setFeedback('正在建立预览列表……');
  const payload = await request('/api/scan/start', { method: 'POST' });
  state.scanId = payload.scan_id;
  const images = filteredAndSortedImages(payload.images || []);
  renderGallery(images, { ...payload, view_count: images.length });
  renderStatus(payload);
  setFeedback(`已显示 ${payload.total} 张预览，正在后台评分……`);
  await pollScan(payload.scan_id);
}

async function pollScan(scanId) {
  while (true) {
    const payload = await request(`/api/scan/${scanId}`);
    const images = filteredAndSortedImages(payload.images || []);
    renderGallery(images, { ...payload, view_count: images.length });
    renderStatus(payload);

    if (payload.done) {
      state.scanActive = false;
      state.scanId = null;
      setScanControlsDisabled(false);
      if (payload.error) {
        setFeedback(payload.error, true);
        return;
      }
      const switched = autoSwitchHiddenView(payload);
      await refreshImages();
      setFeedback(switched ? '扫描完成。当前图片都被模糊规则隐藏，已自动切到未显示图片视图。' : '扫描完成。');
      return;
    }

    setFeedback(`预览已显示 ${payload.total} 张，已评分 ${payload.processed}/${payload.total}……`);
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
}

async function downloadModel() {
  setFeedback('正在下载模型权重，时间取决于模型大小……');
  const payload = await request('/api/models/download', {
    method: 'POST',
    body: JSON.stringify({ model_name: els.modelName.value }),
  });
  state.models = payload.models;
  renderModels();
  setFeedback(`已下载 ${payload.result.model_name} 权重到 ${payload.result.weight_path}`);
}

async function moveToTrash(imageId) {
  const payload = await request(`/api/images/${imageId}/trash`, { method: 'POST' });
  setFeedback(`已移动到回收站：${payload.destination}`);
  await refreshImages();
}

async function emptyTrash() {
  const payload = await request('/api/trash/empty', { method: 'POST' });
  setFeedback(`已清空回收站，删除 ${payload.removed_files} 个文件。`);
}

async function openRecycleBin() {
  const payload = await request('/api/recycle-bin/open', { method: 'POST' });
  setFeedback(`已打开回收站：${payload.path}`);
}

function openViewer(index) {
  if (index < 0 || index >= state.images.length) {
    return;
  }
  state.currentIndex = index;
  const image = state.images[index];
  els.viewer.classList.remove('hidden');
  els.viewer.setAttribute('aria-hidden', 'false');
  els.viewerImage.src = `/api/media/${image.id}?variant=full`;
  const scoreText = image.score == null ? '扫描中' : `${image.score.toFixed(2)} 分`;
  const blurText = image.blur_score == null ? '清晰度待计算' : `清晰度 ${image.blur_score.toFixed(1)}`;
  els.viewerCaption.textContent = `#${image.rank || image.scan_index || index + 1} · ${image.filename} · ${scoreText} · ${blurText}`;
}

function closeViewer() {
  state.currentIndex = -1;
  els.viewer.classList.add('hidden');
  els.viewer.setAttribute('aria-hidden', 'true');
  els.viewerImage.src = '';
}

function shiftViewer(step) {
  if (state.currentIndex === -1 || !state.images.length) {
    return;
  }
  const nextIndex = (state.currentIndex + step + state.images.length) % state.images.length;
  openViewer(nextIndex);
}

document.addEventListener('click', async event => {
  const openIndex = event.target.closest('[data-open]')?.dataset.open;
  const trashId = event.target.closest('[data-trash]')?.dataset.trash;

  if (openIndex !== undefined) {
    openViewer(Number(openIndex));
    return;
  }

  if (trashId) {
    event.stopPropagation();
    await moveToTrash(trashId);
  }
});

document.addEventListener('keydown', async event => {
  if (event.code === 'Space' && state.currentIndex >= 0) {
    event.preventDefault();
    const image = state.images[state.currentIndex];
    if (image) {
      await moveToTrash(image.id);
      closeViewer();
    }
    return;
  }

  if (event.key === 'Escape') {
    closeViewer();
  }

  if (event.key === 'ArrowLeft') {
    shiftViewer(-1);
  }

  if (event.key === 'ArrowRight') {
    shiftViewer(1);
  }
});

els.saveConfigBtn.addEventListener('click', () => saveConfig().catch(error => setFeedback(error.message, true)));
els.scanBtn.addEventListener('click', () => runScan().catch(error => setFeedback(error.message, true)));
els.downloadBtn.addEventListener('click', () => downloadModel().catch(error => setFeedback(error.message, true)));
els.pickImageDirBtn.addEventListener('click', () => pickFolder(els.imageDir).catch(error => setFeedback(error.message, true)));
els.pickRecycleBinDirBtn.addEventListener('click', () => pickFolder(els.recycleBinDir).catch(error => setFeedback(error.message, true)));
els.openRecycleBinBtn.addEventListener('click', () => openRecycleBin().catch(error => setFeedback(error.message, true)));
els.emptyTrashBtn.addEventListener('click', () => {
  if (!window.confirm('确认清空回收站吗？该操作不可恢复。')) {
    return;
  }
  emptyTrash().catch(error => setFeedback(error.message, true));
});
els.viewMode.addEventListener('change', async event => {
  state.viewMode = event.target.value;
  await refreshImages().catch(error => setFeedback(error.message, true));
});
els.sortMode.addEventListener('change', async event => {
  state.sortMode = event.target.value;
  await refreshImages().catch(error => setFeedback(error.message, true));
});
els.viewerClose.addEventListener('click', closeViewer);
els.viewerPrev.addEventListener('click', () => shiftViewer(-1));
els.viewerNext.addEventListener('click', () => shiftViewer(1));
els.modelName.addEventListener('change', renderModels);

refreshStatus().catch(error => setFeedback(error.message, true));
