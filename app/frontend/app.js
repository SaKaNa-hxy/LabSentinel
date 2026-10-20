const $ = (selector) => document.querySelector(selector);
const fileInput = $('#fileInput');
const dropzone = $('#dropzone');
const analyzeBtn = $('#analyzeBtn');
const confidenceRange = $('#confidenceRange');
const reportBtn = $('#reportBtn');
const previewImage = $('#previewImage');
const previewVideo = $('#previewVideo');
const cameraVideo = $('#cameraVideo');
const cameraCanvas = $('#cameraCanvas');
const fileState = { file: null };
let latestAnalysis = null;
let analysisContextId = newAnalysisContextId();
let currentMode = 'image';
let cameraStream = null;
let cameraTimer = null;
let cameraBusy = false;
let cameraEventWindow = [];
let lastAlertKey = '';
let lastAlertTime = 0;
let lastHistoryKey = '';
let lastHistoryTime = 0;
let historyItems = [];
let historySource = 'all';
let historyClearing = false;
let historyLoadVersion = 0;
let agentSessionId = null;
let currentHistoryRecordId = null;
const agentToolLabels = {
  general_chat: '自由对话',
  get_current_analysis: '读取检测事实',
  explain_risk: '解释风险依据',
  search_history: '查询历史记录',
  generate_rectification_plan: '生成整改建议',
  clarify_context: '确认讨论对象'
};
const classMeta = { drinking: ['◒', '饮水', 'BEHAVIOR'], eating: ['◓', '进食', 'BEHAVIOR'], gloves: ['◌', '手套', 'PPE'], goggles: ['◉', '护目镜', 'PPE'], head_mask: ['△', '头套', 'PPE'], lab_coat: ['◇', '实验服', 'PPE'], mask: ['⬡', '口罩', 'PPE'], no_gloves: ['!', '未戴手套', 'RISK'], no_head_mask: ['!', '未戴头套', 'RISK'], no_lab_coat: ['!', '未穿实验服', 'RISK'], no_mask: ['!', '未戴口罩', 'RISK'], no_goggles: ['!', '未戴护目镜', 'RISK'] };
const categoryLabels = { BEHAVIOR: '行为检测', PPE: '防护用品', RISK: '防护缺失', TARGET: '检测目标' };
const modeCopy = {
  image: ['上传现场图像', '将现场图片拖放至此', '支持 JPG、PNG、WEBP · 单张图片不超过 10 MB', '选择图片', '开始安全分析', '分析结果仅作为智能辅助判断，请结合现场巡检与实验室应急预案进行最终决策。'],
  video: ['上传现场视频', '将现场视频拖放至此', '支持 MP4、WEBM、MOV · 最多抽取 40 帧分析', '选择视频', '开始视频分析', '视频分析按时间均匀抽帧，短暂事件仍可能漏检，完成后可生成 DeepSeek 智能报告。'],
  camera: ['摄像头实时检测', '启用本机摄像头', '连续帧复核后显示现场告警，不生成 DeepSeek 报告', '启用摄像头', '实时检测中', '摄像头模式每秒分析一帧；同一风险在最近三帧中出现至少两次才提示告警。']
};

function confidence() {
  return Number(confidenceRange.value) / 100;
}

function newAnalysisContextId() {
  return globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function resetReportState(message = '完成图像分析后，可调用 DeepSeek 生成解释。') {
  latestAnalysis = null;
  analysisContextId = newAnalysisContextId();
  currentHistoryRecordId = null;
  reportBtn.disabled = true;
  $('#reportContent').classList.add('hidden');
  $('#reportState').textContent = message;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(value, max));
}

function resizeAdaptiveFrame(container) {
  if (!container || !container.dataset.mediaWidth || !container.dataset.mediaHeight) return;
  const mediaWidth = Number(container.dataset.mediaWidth);
  const mediaHeight = Number(container.dataset.mediaHeight);
  if (!mediaWidth || !mediaHeight) return;
  if (container.classList.contains('video-frame')) {
    container.style.aspectRatio = `${mediaWidth} / ${mediaHeight}`;
    container.style.height = '';
    return;
  }
  const frameWidth = container.getBoundingClientRect().width || container.parentElement.getBoundingClientRect().width;
  if (!frameWidth) return;
  const isResult = container.id === 'resultImageWrap';
  const minHeight = isResult ? 300 : 250;
  const maxHeight = window.innerHeight * (isResult ? 0.78 : 0.72);
  const idealHeight = frameWidth * (mediaHeight / mediaWidth);
  const height = clamp(idealHeight, minHeight, maxHeight);
  container.style.height = `${Math.round(height)}px`;
}

function setAdaptiveFrame(container, width, height) {
  if (!container || !width || !height) return;
  container.dataset.mediaWidth = String(width);
  container.dataset.mediaHeight = String(height);
  container.classList.add('adaptive-frame');
  requestAnimationFrame(() => resizeAdaptiveFrame(container));
}

function clearAdaptiveFrame(container) {
  if (!container) return;
  delete container.dataset.mediaWidth;
  delete container.dataset.mediaHeight;
  container.style.height = '';
  container.style.aspectRatio = '';
  container.classList.remove('video-frame');
  container.classList.remove('adaptive-frame');
}

function setMode(mode) {
  stopCamera();
  currentMode = mode;
  fileState.file = null;
  fileInput.value = '';
  clearAdaptiveFrame(previewImage.closest('.preview-frame'));
  clearAdaptiveFrame($('#resultImageWrap'));
  fileInput.accept = mode === 'video' ? 'video/mp4,video/webm,video/quicktime' : 'image/png,image/jpeg,image/webp';
  document.querySelectorAll('.mode-tab').forEach((button) => {
    const selected = button.dataset.mode === mode;
    button.classList.toggle('active', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
  const copy = modeCopy[mode];
  $('#workspaceTitle').textContent = copy[0];
  $('#uploadTitle').textContent = copy[1];
  $('#uploadHint').textContent = copy[2];
  $('#browseBtn').textContent = copy[3];
  analyzeBtn.querySelector('span').textContent = copy[4];
  $('#modeNotice').textContent = copy[5];
  $('#emptyState').classList.toggle('hidden', mode === 'camera');
  $('#previewState').classList.add('hidden');
  $('#cameraState').classList.toggle('hidden', mode !== 'camera');
  previewImage.src = '';
  previewVideo.src = '';
  previewVideo.onloadedmetadata = null;
  previewImage.classList.remove('hidden');
  previewVideo.classList.add('hidden');
  analyzeBtn.disabled = mode !== 'camera';
  resetReportState(mode === 'camera' ? '摄像头实时检测不生成智能报告。' : '完成分析后，可调用 DeepSeek 生成解释。');
}

function setFile(file) {
  if (!file) return;
  const isImage = currentMode === 'image' && file.type.startsWith('image/');
  const isVideo = currentMode === 'video' && file.type.startsWith('video/');
  if (!isImage && !isVideo) return;
  if (isImage && file.size > 10 * 1024 * 1024) {
    $('#riskSummary').textContent = '图片不能超过 10 MB，请选择较小的图片。';
    return;
  }
  fileState.file = file;
  resetReportState('完成分析后，可调用 DeepSeek 生成解释。');
  $('#fileName').textContent = file.name;
  $('#emptyState').classList.add('hidden');
  $('#previewState').classList.remove('hidden');
  analyzeBtn.disabled = false;
  clearAdaptiveFrame(previewImage.closest('.preview-frame'));
  if (isVideo) {
    const frame = previewVideo.closest('.preview-frame');
    frame.classList.add('video-frame');
    previewVideo.onloadedmetadata = () => setAdaptiveFrame(frame, previewVideo.videoWidth, previewVideo.videoHeight);
    previewVideo.src = URL.createObjectURL(file);
    previewVideo.classList.remove('hidden');
    previewImage.classList.add('hidden');
  } else {
    previewImage.onload = () => setAdaptiveFrame(previewImage.closest('.preview-frame'), previewImage.naturalWidth, previewImage.naturalHeight);
    previewImage.src = URL.createObjectURL(file);
    previewImage.classList.remove('hidden');
    previewVideo.classList.add('hidden');
  }
}

function clearFile() {
  fileState.file = null;
  fileInput.value = '';
  clearAdaptiveFrame(previewImage.closest('.preview-frame'));
  previewImage.onload = null;
  previewImage.src = '';
  previewVideo.src = '';
  previewVideo.onloadedmetadata = null;
  $('#previewState').classList.add('hidden');
  $('#emptyState').classList.remove('hidden');
  analyzeBtn.disabled = true;
  resetReportState(currentMode === 'camera' ? '摄像头实时检测不生成智能报告。' : '完成分析后，可调用 DeepSeek 生成解释。');
}

$('#browseBtn').addEventListener('click', () => {
  if (currentMode === 'camera') {
    startCamera();
    return;
  }
  fileInput.click();
});
fileInput.addEventListener('change', (event) => setFile(event.target.files[0]));
$('#removeBtn').addEventListener('click', clearFile);
$('.mode-tabs').addEventListener('click', (event) => {
  const button = event.target.closest('.mode-tab');
  if (button) setMode(button.dataset.mode);
});
['dragenter', 'dragover'].forEach((eventName) => dropzone.addEventListener(eventName, (event) => { event.preventDefault(); if (currentMode !== 'camera') dropzone.classList.add('dragging'); }));
['dragleave', 'drop'].forEach((eventName) => dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.remove('dragging'); }));
dropzone.addEventListener('drop', (event) => setFile(event.dataTransfer.files[0]));
confidenceRange.addEventListener('input', (event) => { const value = event.target.value; $('#rangeOutput').textContent = `${value}%`; $('#confidenceValue').textContent = (value / 100).toFixed(2); });
window.addEventListener('resize', () => {
  resizeAdaptiveFrame(previewImage.closest('.preview-frame'));
  resizeAdaptiveFrame($('#resultImageWrap'));
});

function renderSignals(counts = {}) {
  $('#signalGrid').innerHTML = Object.entries(classMeta).map(([key, meta]) => `<div class="signal-card" data-category="${meta[2]}"><span class="signal-icon" aria-hidden="true">${meta[0]}</span><b>${meta[1]}</b><small>${categoryLabels[meta[2]]} / <strong>${counts[key] || 0}</strong> 个目标</small></div>`).join('');
}

function renderRisk(risk) {
  const events = risk.events || [];
  const hasEvents = events.length > 0;
  $('#eventStatus').textContent = hasEvents ? `${events.length} 类风险提示` : '暂无风险事件';
  $('#eventList').innerHTML = hasEvents
    ? events.map((event) => `<span class="event-badge event-${event.key}"><i></i>${event.label}</span>`).join('')
    : '<span class="event-empty">暂未发现进食、饮水或 PPE 风险</span>';
  $('#riskSummary').textContent = risk.summary;
  $('#recommendationList').innerHTML = risk.recommendations.map((item) => `<li>${item}</li>`).join('');
}

function renderDetections(detections = []) {
  if (!detections.length) {
    $('#detectionList').innerHTML = '<div class="detection-empty">未检测到目标，可降低置信度或换一张更清晰的图片复核</div>';
    return;
  }
  $('#detectionList').innerHTML = detections.map((item, index) => {
    const meta = classMeta[item.class_name] || ['•', item.class_name || '未知目标', 'TARGET'];
    const confidenceText = `${Math.round(Number(item.confidence || 0) * 100)}%`;
    const bboxText = Array.isArray(item.bbox) ? item.bbox.map((value) => Number(value).toFixed(0)).join(', ') : '—';
    const frameText = item.frame ? `<span>第 ${escapeHtml(item.frame)} 帧</span>` : '';
    const riskClass = meta[2] === 'RISK' || meta[2] === 'BEHAVIOR' ? ' detection-item-risk' : '';
    return `<article class="detection-item${riskClass}"><div class="detection-main"><strong>${index + 1}</strong><div><b>${escapeHtml(meta[1])}</b><small>${escapeHtml(categoryLabels[meta[2]] || meta[2])} ${frameText}</small></div></div><div class="detection-meta"><span>${confidenceText}</span><small>坐标 ${escapeHtml(bboxText)}</small></div></article>`;
  }).join('');
}

async function loadModelInfo() {
  try {
    const response = await fetch('/api/model-info');
    const info = await response.json();
    $('#modelStatus').textContent = info.loaded ? (info.custom_model_ready ? '专用模型已加载' : '通用权重 · 待训练专用模型') : '依赖未安装或模型未加载';
  } catch { $('#modelStatus').textContent = '本地服务连接待建立'; }
}

async function analyzeBlob(blob) {
  const formData = new FormData();
  formData.append('file', blob, 'frame.jpg');
  const response = await fetch(`/api/analyze?confidence=${confidence()}`, { method: 'POST', body: formData });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || '分析失败');
  return data;
}

async function saveHistory(source, blob, analysis, frameIndex = null) {
  if (!blob || !analysis) return null;
  const formData = new FormData();
  const imageBlob = analysis.image ? await (await fetch(analysis.image)).blob() : blob;
  formData.append('file', imageBlob, 'annotated.jpg');
  formData.append('source', source);
  formData.append('analysis', JSON.stringify({ risk: analysis.risk, detections: analysis.detections, warning: analysis.warning, image_size: analysis.image_size }));
  if (frameIndex !== null && frameIndex !== undefined) formData.append('frame_index', String(frameIndex));
  const response = await fetch('/api/history', { method: 'POST', body: formData });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || '历史记录保存失败');
  historyItems = [data, ...historyItems.filter((item) => item.id !== data.id)];
  renderHistory();
  return data;
}

async function analyzeImage() {
  const data = await analyzeBlob(fileState.file);
  latestAnalysis = data;
  const record = await saveHistory('image', fileState.file, data);
  currentHistoryRecordId = record && record.id;
  reportBtn.disabled = false;
  $('#reportContent').classList.add('hidden');
  $('#reportState').textContent = '检测结果已就绪，点击按钮生成 DeepSeek 智能报告。';
  renderAnalysis(data, `刚刚完成 · ${data.image_size.width} × ${data.image_size.height}`);
}

function captureVideoFrame(video) {
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.88));
}

function seekVideo(video, time) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      video.removeEventListener('seeked', done);
      reject(new Error('视频帧读取超时'));
    }, 6000);
    const done = () => {
      clearTimeout(timer);
      video.removeEventListener('seeked', done);
      resolve();
    };
    video.addEventListener('seeked', done, { once: true });
    video.currentTime = time;
  });
}

function hasRisk(data) {
  return Boolean(data && data.risk && (data.risk.events || []).length);
}

function mergeVideoResults(results) {
  const counts = { drinking: 0, eating: 0, gloves: 0, goggles: 0, head_mask: 0, lab_coat: 0, mask: 0, no_gloves: 0, no_head_mask: 0, no_lab_coat: 0, no_mask: 0, no_goggles: 0 };
  const eventMap = new Map();
  const detections = [];
  const missingPpe = [];
  results.forEach((result, index) => {
    detections.push(...result.detections.map((item) => ({ ...item, frame: index + 1 })));
    Object.entries(result.risk.counts || {}).forEach(([key, value]) => { counts[key] = Math.max(counts[key] || 0, value); });
    (result.risk.events || []).forEach((event) => eventMap.set(event.key, event));
    (result.risk.missing_ppe || []).forEach((item) => missingPpe.push({ ...item, frame: index + 1 }));
  });
  const events = [...eventMap.values()];
  const recommendations = events.length
    ? ['复核视频中出现风险的关键帧', '对进食、饮水和 PPE 风险进行现场确认', '保留视频分析结果作为巡检记录']
    : ['继续保持实验区域整洁和通风', '建议结合更多角度视频复核检测结果', '分析结果仅作为安全辅助判断'];
  const risk = {
    title: events.length ? '视频中发现风险事件' : '视频中未发现明确安全事件',
    summary: events.length ? `视频关键帧分析发现 ${events.length} 类风险事件。` : '已完成视频关键帧分析，暂未发现进食、饮水或 PPE 风险。',
    events,
    detected_hazards: counts.eating + counts.drinking,
    missing_ppe: missingPpe,
    recommendations,
    counts,
    counts_semantics: 'maximum_per_frame',
    confidence_by_class: {}
  };
  const representative = results.find(hasRisk) || results[results.length - 1];
  return { ...representative, detections, risk, frame_count: results.length };
}

async function analyzeVideo() {
  const video = previewVideo;
  if (!video.duration || !video.videoWidth) await new Promise((resolve) => video.addEventListener('loadedmetadata', resolve, { once: true }));
  const duration = video.duration || 1;
  const frameCount = Math.min(40, Math.max(3, Math.ceil(duration)));
  const results = [];
  for (let index = 0; index < frameCount; index += 1) {
    const time = Math.min(duration - 0.1, (duration * (index + 0.5)) / frameCount);
    $('#analysisTag').textContent = `第 ${index + 1} / ${frameCount} 帧`;
    await seekVideo(video, Math.max(0, time));
    const blob = await captureVideoFrame(video);
    const frameAnalysis = await analyzeBlob(blob);
    results.push({ blob, data: frameAnalysis, frameIndex: index + 1 });
  }
  const data = mergeVideoResults(results.map((item) => item.data));
  latestAnalysis = data;
  let latestRecord = null;
  for (const frame of results) {
    latestRecord = await saveHistory('video', frame.blob, frame.data, frame.frameIndex);
  }
  currentHistoryRecordId = latestRecord && latestRecord.id;
  reportBtn.disabled = false;
  $('#reportContent').classList.add('hidden');
  $('#reportState').textContent = '视频关键帧分析已就绪，点击按钮生成 DeepSeek 智能报告。';
  renderAnalysis(data, `视频分析完成 · ${data.frame_count} 个关键帧`);
}

function renderAnalysis(data, timeText) {
  const imageSize = data.image_size || {};
  if (imageSize.width && imageSize.height) setAdaptiveFrame($('#resultImageWrap'), imageSize.width, imageSize.height);
  $('#resultImage').onload = () => setAdaptiveFrame($('#resultImageWrap'), $('#resultImage').naturalWidth, $('#resultImage').naturalHeight);
  $('#resultImage').src = data.image;
  $('#resultImage').classList.remove('hidden');
  $('.result-placeholder').classList.add('hidden');
  $('#analysisTag').textContent = `${data.detections.length} TARGETS FOUND`;
  $('#resultTime').textContent = timeText;
  renderRisk(data.risk);
  if (data.warning) $('#riskSummary').textContent += ` 模型提示：${data.warning}`;
  renderDetections(data.detections);
  renderSignals(data.risk.counts);
  if (currentMode !== 'camera') document.querySelector('#resultsSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function alertFromRisk(risk) {
  const events = risk.events || [];
  const keys = events.map((event) => event.key).sort().join('|');
  const now = Date.now();
  const banner = $('#liveAlert');
  if (!keys) {
    banner.classList.add('hidden');
    lastAlertKey = '';
    return;
  }
  if (keys === lastAlertKey && now - lastAlertTime < 10000) return;
  lastAlertKey = keys;
  lastAlertTime = now;
  const labels = events.map((event) => event.label).join('、');
  banner.textContent = `实验室安全提示：${labels}。请立即现场复核。`;
  banner.classList.remove('hidden');
}

async function cameraTick() {
  if (!cameraStream || cameraBusy || !cameraVideo.videoWidth) return;
  cameraBusy = true;
  try {
    cameraCanvas.width = cameraVideo.videoWidth;
    cameraCanvas.height = cameraVideo.videoHeight;
    cameraCanvas.getContext('2d').drawImage(cameraVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);
    const blob = await new Promise((resolve) => cameraCanvas.toBlob(resolve, 'image/jpeg', 0.82));
    const data = await analyzeBlob(blob);
    if (!cameraStream) return;
    latestAnalysis = data;
    renderAnalysis(data, `摄像头实时检测 · ${new Date().toLocaleTimeString()}`);
    reportBtn.disabled = true;
    $('#reportState').textContent = '摄像头实时检测不生成智能报告。';
    const events = data.risk.events || [];
    cameraEventWindow.push(new Set(events.map((event) => event.key)));
    if (cameraEventWindow.length > 3) cameraEventWindow.shift();
    const stableEvents = events.filter((event) => cameraEventWindow.filter((frame) => frame.has(event.key)).length >= 2);
    const stableRisk = { events: stableEvents };
    const keys = stableEvents.map((event) => event.key).sort().join('|');
    const now = Date.now();
    alertFromRisk(stableRisk);
    if (keys && (keys !== lastHistoryKey || now - lastHistoryTime >= 10000)) {
      lastHistoryKey = keys;
      lastHistoryTime = now;
      const record = await saveHistory('camera', blob, data);
      currentHistoryRecordId = record && record.id;
    }
  } catch (error) {
    $('#analysisTag').textContent = 'CAMERA ERROR';
    $('#riskSummary').textContent = error.message;
  } finally {
    cameraBusy = false;
  }
}

async function startCamera() {
  if (cameraStream) return;
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    cameraVideo.srcObject = cameraStream;
    await cameraVideo.play();
    resetReportState('摄像头实时检测不生成智能报告。');
    $('#startCameraBtn').disabled = true;
    $('#stopCameraBtn').disabled = false;
    $('#analysisTag').textContent = 'CAMERA LIVE';
    cameraEventWindow = [];
    lastAlertKey = '';
    lastHistoryKey = '';
    cameraTimer = setInterval(cameraTick, 1000);
    cameraTick();
  } catch (error) {
    $('#riskSummary').textContent = `摄像头启用失败：${error.message}`;
  }
}

function stopCamera() {
  if (cameraTimer) clearInterval(cameraTimer);
  cameraTimer = null;
  if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  cameraBusy = false;
  cameraEventWindow = [];
  $('#liveAlert').classList.add('hidden');
  $('#startCameraBtn').disabled = false;
  $('#stopCameraBtn').disabled = true;
}

analyzeBtn.addEventListener('click', async () => {
  if (currentMode === 'camera') {
    startCamera();
    return;
  }
  if (!fileState.file) return;
  resetReportState('正在分析当前文件…');
  analyzeBtn.disabled = true;
  analyzeBtn.querySelector('span').textContent = currentMode === 'video' ? '分析中 · 正在抽帧' : '分析中 · 正在扫描';
  $('#analysisTag').textContent = 'PROCESSING';
  try {
    if (currentMode === 'video') await analyzeVideo();
    else await analyzeImage();
  } catch (error) {
    $('#analysisTag').textContent = 'ANALYSIS ERROR';
    $('#riskSummary').textContent = error.message;
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.querySelector('span').textContent = currentMode === 'video' ? '开始视频分析' : '开始安全分析';
  }
});

$('#startCameraBtn').addEventListener('click', startCamera);
$('#stopCameraBtn').addEventListener('click', stopCamera);

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'\"]/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '\"': '&quot;' })[character]);
}

function formatHistoryTime(value) {
  if (!value) return '未知时间';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN', { hour12: false });
}

function renderHistory() {
  $('#clearHistoryBtn').disabled = historyClearing || historyItems.length === 0;
  $('#clearHistoryBtn').textContent = historyClearing ? '正在删除…' : '删除全部记录';
  const visibleItems = historyItems.filter((item) => historySource === 'all' || item.source === historySource);
  $('#historyCount').textContent = `${visibleItems.length} 条记录`;
  if (!visibleItems.length) {
    $('#historyGrid').innerHTML = '<div class="history-empty">暂无符合条件的历史分析记录</div>';
    return;
  }
  $('#historyGrid').innerHTML = visibleItems.map((item) => {
    const risk = item.risk || {};
    const events = (risk.events || []).map((event) => escapeHtml(event.label)).join('、') || '未发现明确风险';
    const frame = item.frame_index ? `<span>第 ${escapeHtml(item.frame_index)} 帧</span>` : '';
    const riskClass = (risk.events || []).length ? 'history-risk warning' : 'history-risk safe';
    return `<article class="history-card" data-preview-id="${escapeHtml(item.id)}"><button class="history-preview-btn" type="button" aria-label="放大查看${escapeHtml(item.source_label)}图片，${escapeHtml(formatHistoryTime(item.created_at))}"><img class="history-thumb" src="${escapeHtml(item.image_url)}" alt="${escapeHtml(item.source_label)}截图"><span class="history-zoom-hint" aria-hidden="true">查看大图 ⤢</span></button><div class="history-card-body"><div class="history-card-top"><span class="history-source source-${escapeHtml(item.source)}">${escapeHtml(item.source_label)}</span><button class="history-delete" data-history-id="${escapeHtml(item.id)}" type="button">删除</button></div><div class="history-time">${escapeHtml(formatHistoryTime(item.created_at))} ${frame}</div><h3>${escapeHtml(risk.title || '分析结果')}</h3><p class="${riskClass}">${events}</p></div></article>`;
  }).join('');
}

async function loadHistory() {
  const version = ++historyLoadVersion;
  try {
    const response = await fetch('/api/history');
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '历史记录加载失败');
    if (version !== historyLoadVersion) return false;
    historyItems = Array.isArray(data.items) ? data.items : [];
    renderHistory();
    return true;
  } catch (error) {
    if (version !== historyLoadVersion) return false;
    $('#historyGrid').innerHTML = `<div class="history-empty">历史记录加载失败：${escapeHtml(error.message)}</div>`;
    return false;
  }
}

async function removeHistory(id) {
  if (!window.confirm('确定删除这条历史分析记录及对应图片吗？')) return;
  const response = await fetch(`/api/history/${encodeURIComponent(id)}`, { method: 'DELETE' });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || '删除失败');
  historyItems = historyItems.filter((item) => item.id !== id);
  if (currentHistoryRecordId === id) currentHistoryRecordId = null;
  renderHistory();
}

async function clearHistory() {
  if (historyClearing || !historyItems.length) return;
  historyClearing = true;
  ++historyLoadVersion;
  $('#refreshHistoryBtn').disabled = true;
  $('#historyActionStatus').textContent = '正在删除全部历史记录…';
  renderHistory();
  try {
    const response = await fetch('/api/history', { method: 'DELETE' });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '全部删除失败');
    const refreshed = await loadHistory();
    if (refreshed && !historyItems.some((item) => item.id === currentHistoryRecordId)) currentHistoryRecordId = null;
    $('#historyActionStatus').textContent = data.failed_count
      ? `已删除 ${data.deleted_count} 条，${data.failed_count} 条删除失败，请检查文件是否被占用后重试。`
      : `已删除 ${data.deleted_count} 条历史记录及对应截图。${refreshed ? '' : '列表刷新失败，请点击刷新记录。'}`;
  } catch (error) {
    $('#historyActionStatus').textContent = `删除失败：${error.message}`;
    await loadHistory();
  } finally {
    historyClearing = false;
    $('#refreshHistoryBtn').disabled = false;
    if ($('#historyGrid .history-empty')?.textContent.startsWith('历史记录加载失败')) {
      $('#clearHistoryBtn').disabled = false;
      $('#clearHistoryBtn').textContent = '删除全部记录';
    } else {
      renderHistory();
    }
  }
}

$('#clearHistoryBtn').addEventListener('click', () => {
  if (historyClearing || !historyItems.length) return;
  $('#clearHistoryDescription').textContent = `将删除全部 ${historyItems.length} 条历史分析记录及对应截图，包含图片、视频和摄像头记录，不限当前筛选类型。此操作无法撤销。${cameraStream ? '摄像头仍在检测，删除完成后可能继续产生新记录。' : ''}`;
  $('#clearHistoryDialog').showModal();
});
$('#cancelClearHistoryBtn').addEventListener('click', () => $('#clearHistoryDialog').close());
$('#confirmClearHistoryBtn').addEventListener('click', () => {
  $('#clearHistoryDialog').close();
  clearHistory();
});
$('#refreshHistoryBtn').addEventListener('click', loadHistory);
$('#historyFilters').addEventListener('click', (event) => {
  const button = event.target.closest('.history-filter');
  if (!button) return;
  historySource = button.dataset.source;
  document.querySelectorAll('.history-filter').forEach((item) => item.classList.toggle('active', item === button));
  renderHistory();
});
const historyImageDialog = $('#historyImageDialog');
const historyFullImage = $('#historyFullImage');
let historyPreviewTrigger = null;

function openHistoryImage(id, trigger) {
  const record = historyItems.find((item) => item.id === id);
  if (!record) return;
  historyPreviewTrigger = trigger;
  $('#historyImageTitle').textContent = record.risk?.title || '历史记录图片';
  $('#historyImageMeta').textContent = `${record.source_label} / ${formatHistoryTime(record.created_at)}${record.frame_index ? ` / 第 ${record.frame_index} 帧` : ''}`;
  $('#historyImageError').classList.add('hidden');
  historyFullImage.classList.remove('hidden');
  historyFullImage.alt = `${record.source_label}的历史标注图片`;
  historyFullImage.src = record.image_url;
  historyImageDialog.showModal();
  document.body.classList.add('history-viewer-open');
}

historyFullImage.addEventListener('error', () => {
  if (!historyImageDialog.open) return;
  historyFullImage.classList.add('hidden');
  $('#historyImageError').classList.remove('hidden');
});
$('#closeHistoryImageBtn').addEventListener('click', () => historyImageDialog.close());
historyImageDialog.addEventListener('click', (event) => {
  if (event.target !== historyImageDialog) return;
  const bounds = historyImageDialog.getBoundingClientRect();
  if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) historyImageDialog.close();
});
historyImageDialog.addEventListener('close', () => {
  document.body.classList.remove('history-viewer-open');
  historyFullImage.removeAttribute('src');
  if (historyPreviewTrigger?.isConnected) historyPreviewTrigger.focus({ preventScroll: true });
  historyPreviewTrigger = null;
});

$('#historyGrid').addEventListener('click', async (event) => {
  const button = event.target.closest('.history-delete');
  if (historyClearing) return;
  if (!button) {
    const card = event.target.closest('.history-card');
    if (card) openHistoryImage(card.dataset.previewId, card.querySelector('.history-preview-btn'));
    return;
  }
  button.disabled = true;
  try {
    await removeHistory(button.dataset.historyId);
  } catch (error) {
    window.alert(error.message);
  } finally {
    button.disabled = false;
  }
});

reportBtn.addEventListener('click', async () => {
  if (!latestAnalysis || currentMode === 'camera') return;
  reportBtn.disabled = true;
  reportBtn.querySelector('strong').textContent = '…';
  $('#reportState').textContent = 'DeepSeek 正在整理检测事实与处置建议…';
  $('#reportContent').classList.add('hidden');
  try {
    const response = await fetch('/api/report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ detections: latestAnalysis.detections, risk: latestAnalysis.risk })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '报告生成失败');
    $('#reportContent').textContent = data.report;
    $('#reportContent').classList.remove('hidden');
    $('#reportState').textContent = `${data.provider} · ${data.model} 已完成分析`;
  } catch (error) {
    $('#reportState').textContent = error.message;
  } finally {
    reportBtn.disabled = false;
    reportBtn.querySelector('strong').textContent = '↗';
  }
});

function appendAgentMessage(role, content) {
  const message = document.createElement('div');
  message.className = `agent-message agent-message-${role}`;
  const label = document.createElement('b');
  label.textContent = role === 'user' ? 'YOU' : 'LABSENTINEL AGENT';
  const body = document.createElement('p');
  body.textContent = content;
  message.append(label, body);
  $('#agentMessages').appendChild(message);
  $('#agentMessages').scrollTop = $('#agentMessages').scrollHeight;
}

async function sendAgentMessage(message) {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      session_id: agentSessionId,
      record_id: currentHistoryRecordId,
      analysis_id: analysisContextId,
      analysis: latestAnalysis ? { detections: latestAnalysis.detections, risk: latestAnalysis.risk, frame_count: latestAnalysis.frame_count, source_type: currentMode } : null
    })
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'Agent 请求失败');
  agentSessionId = data.session_id;
  return data;
}

$('#agentForm').addEventListener('submit', async (event) => {
  event.preventDefault();
  const input = $('#agentInput');
  const message = input.value.trim();
  if (!message) return;
  const button = $('#agentSendBtn');
  input.value = '';
  input.disabled = true;
  button.disabled = true;
  appendAgentMessage('user', message);
  $('#agentState').textContent = 'Agent 正在理解问题并选择处理方式…';
  try {
    const data = await sendAgentMessage(message);
    appendAgentMessage('bot', data.answer);
    const focus = data.tool === 'search_history' ? null : data.discussion_focus;
    const focusParts = [...(focus?.frames || []).map((frame) => `第${frame}帧`), ...(focus?.categories || [])];
    $('#agentState').textContent = [agentToolLabels[data.tool] || data.tool, ...focusParts, data.provider].join(' · ');
  } catch (error) {
    appendAgentMessage('bot', error.message);
    $('#agentState').textContent = '请求失败，请稍后重试';
  } finally {
    input.disabled = false;
    button.disabled = false;
    input.focus();
  }
});

$('#clearChatBtn').addEventListener('click', async () => {
  if (agentSessionId) {
    await fetch('/api/chat/clear', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: agentSessionId })
    });
  }
  agentSessionId = null;
  $('#agentMessages').innerHTML = '<div class="agent-message agent-message-bot"><b>LabSentinel Agent</b><p>会话已清空。你可以和我自由聊天，也可以让我解释风险依据、查询历史记录或给出整改建议。</p></div>';
  $('#agentState').textContent = '可自由聊天，也可询问当前检测、风险依据、历史记录或整改建议';
});

renderSignals();
loadModelInfo();
loadHistory();
