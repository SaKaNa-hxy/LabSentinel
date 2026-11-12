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
let analysisContextId = newLocalId();
let currentMode = 'image';
let cameraStream = null;
let cameraTimer = null;
let cameraBusy = false;
let cameraFrameIndex = 0;
let cameraEventWindow = [];
const cameraConfirmedEvents = new Map();
let lastHistoryKey = '';
let lastHistoryTime = 0;
let historyItems = [];
let historySource = 'all';
let historyClearing = false;
let historyLoadVersion = 0;
let historyLoaded = false;
let chatItems = [];
let chatListVersion = 0;
let agentSessionId = null;
let agentBusy = false;
let agentAttachmentIds = [];
let agentUseCurrentAnalysis = true;
let agentHasSavedAnalysis = false;
let agentRetry = null;
let currentHistoryRecordId = null;
const maxAgentAttachments = 4;
const activeChatStorageKey = 'labsentinel.activeChat';
const historyDragType = 'application/x-labsentinel-history-id';
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
  camera: ['摄像头实时检测', '启用本机摄像头', '同一风险在最近三帧中至少出现两次时确认并保存', '启用摄像头', '实时检测中', '本次检测已确认的风险保留在「风险复核」，截图可在「分析历史」中查看。']
};

function confidence() {
  return Number(confidenceRange.value) / 100;
}

function newLocalId() {
  return globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function resetReportState(message = '完成图像分析后，可调用 DeepSeek 生成解释。') {
  latestAnalysis = null;
  analysisContextId = newLocalId();
  currentHistoryRecordId = null;
  reportBtn.disabled = true;
  reportBtn.querySelector('strong').textContent = '↗';
  $('#reportContent').classList.add('hidden');
  $('#reportState').textContent = message;
  updateAgentContext();
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
window.addEventListener('resize', resizeDetectionFrames);

function renderSignals(counts = {}, risk = {}) {
  const pendingCoat = (risk.pending_events || []).some(event => event.key === 'no_lab_coat') && !(risk.events || []).some(event => event.key === 'no_lab_coat');
  $('#signalGrid').innerHTML = Object.entries(classMeta).map(([key, meta]) => `<div class="signal-card" data-category="${meta[2]}"><span class="signal-icon" aria-hidden="true">${meta[0]}</span><b>${meta[1]}</b><small>${key === 'no_lab_coat' && pendingCoat ? '待复核检测' : categoryLabels[meta[2]]} / <strong>${counts[key] || 0}</strong> 个目标</small></div>`).join('');
}

function renderRisk(risk) {
  const events = risk.events || [];
  const hasEvents = events.length > 0;
  $('#eventStatus').textContent = hasEvents ? `${events.length} 类风险提示` : '暂无风险事件';
  $('#eventList').innerHTML = hasEvents
    ? events.map((event) => `<span class="event-badge event-${event.key}"><i></i>${event.label}</span>`).join('')
    : '<span class="event-empty">暂未发现进食、饮水或 PPE 风险</span>';
  if ((risk.pending_events || []).length) {
    if (!hasEvents) $('#eventList').innerHTML = '';
    $('#eventList').innerHTML += risk.pending_events.map(event => `<span class="event-pending">待复核：${escapeHtml(event.label)}</span>`).join('');
  }
  $('#riskSummary').textContent = risk.summary;
  $('#recommendationList').innerHTML = risk.recommendations.map((item) => `<li>${item}</li>`).join('');
}

function renderDetections(detections = [], risk = {}) {
  if (!detections.length) {
    $('#detectionList').innerHTML = '<div class="detection-empty">未检测到目标，可降低置信度或换一张更清晰的图片复核</div>';
    return;
  }
  const pendingFrames = new Set((((risk.temporal_confirmation || {}).no_lab_coat || {}).pending_frames || []));
  $('#detectionList').innerHTML = detections.map((item, index) => {
    const meta = classMeta[item.class_name] || ['•', item.class_name || '未知目标', 'TARGET'];
    const confidenceText = `${Math.round(Number(item.confidence || 0) * 100)}%`;
    const bboxText = Array.isArray(item.bbox) ? item.bbox.map((value) => Number(value).toFixed(0)).join(', ') : '—';
    const pending = item.class_name === 'no_lab_coat' && pendingFrames.has(item.frame);
    const frameText = (item.frame ? `<span>第 ${escapeHtml(item.frame)} 帧</span>` : '') + (pending ? '<span>待复核，未满足连续2帧</span>' : '');
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
  formData.append('analysis', JSON.stringify({ risk: analysis.risk, detections: analysis.detections, warning: analysis.warning, image_size: analysis.image_size, postprocessing: analysis.postprocessing, video_context: analysis.video_context }));
  if (frameIndex !== null && frameIndex !== undefined) formData.append('frame_index', String(frameIndex));
  const response = await fetch('/api/history', { method: 'POST', body: formData });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || '历史记录保存失败');
  historyItems = [data, ...historyItems.filter((item) => item.id !== data.id)];
  renderHistory();
  if (historyPicker.open) renderHistoryPicker();
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

function videoCoatConfirmation(detections) {
  const detected = [...new Set(detections.filter(item => item.class_name === 'no_lab_coat' && Number.isInteger(item.frame) && item.frame > 0).map(item => item.frame))].sort((a, b) => a - b);
  const frames = new Set(detected);
  const confirmed = detected.filter(frame => frames.has(frame - 1) || frames.has(frame + 1));
  const confirmedSet = new Set(confirmed);
  return { minimum_consecutive_frames: 2, detected_frames: detected, confirmed_frames: confirmed, pending_frames: detected.filter(frame => !confirmedSet.has(frame)), unit: 'sampled_keyframe', person_tracking: 'not_available' };
}

function applyVideoCoatConfirmation(risk, evidence, selectedFrames = null) {
  const selected = selectedFrames ? new Set(selectedFrames) : null;
  const relevant = frames => frames.filter(frame => !selected || selected.has(frame));
  const detected = relevant(evidence.detected_frames);
  const confirmed = relevant(evidence.confirmed_frames);
  const pending = relevant(evidence.pending_frames);
  const supporting = new Set();
  const allFrames = new Set(evidence.detected_frames);
  confirmed.forEach(frame => {
    let start = frame, end = frame;
    while (allFrames.has(start - 1)) start -= 1;
    while (allFrames.has(end + 1)) end += 1;
    for (let index = start; index <= end; index += 1) supporting.add(index);
  });
  const original = (risk.events || []).find(event => event.key === 'no_lab_coat');
  const events = (risk.events || []).filter(event => event.key !== 'no_lab_coat');
  const pendingEvents = (risk.pending_events || []).filter(event => event.key !== 'no_lab_coat');
  if (confirmed.length) events.push({ ...(original || {}), key: 'no_lab_coat', label: '未穿实验服（连续多帧检出）', source: 'visual_detection', confirmation: 'consecutive_sampled_frames', evidence_frames: [...supporting].sort((a, b) => a - b) });
  if (pending.length) pendingEvents.push({ key: 'no_lab_coat', label: '未穿实验服（未连续检出）', source: 'visual_detection', status: 'pending_temporal_confirmation', evidence_frames: pending });
  const missingPpe = (risk.missing_ppe || []).filter(item => item.class_name !== 'no_lab_coat' || (confirmed.length && (item.frame === undefined || confirmed.includes(item.frame))));
  const prefix = selected ? '当前采样帧' : '视频关键帧分析';
  const pendingNote = pending.length ? ` 未穿实验服仅在第 ${pending.join('、')} 个采样关键帧出现提示，未满足连续2帧检出条件，列为待复核。` : '';
  return {
    ...risk,
    title: events.length ? `${prefix}风险提示` : (pending.length ? '实验服状态待复核' : '暂未发现明确安全事件'),
    summary: (events.length ? `${prefix}保留 ${events.length} 类风险提示。` : `${prefix}暂未形成确认的风险提示。`) + pendingNote,
    events, pending_events: pendingEvents, missing_ppe: missingPpe,
    recommendations: !events.length && pending.length ? ['复核待确认关键帧及相邻画面的实际衣物', '结合其他角度确认衣领、袖子和衣服款式'] : risk.recommendations,
    temporal_confirmation: { ...(risk.temporal_confirmation || {}), no_lab_coat: { ...evidence, detected_frames: detected, confirmed_frames: confirmed, pending_frames: pending, supporting_frames: [...supporting].sort((a, b) => a - b) } }
  };
}

function mergeVideoResults(results) {
  const counts = { drinking: 0, eating: 0, gloves: 0, goggles: 0, head_mask: 0, lab_coat: 0, mask: 0, no_gloves: 0, no_head_mask: 0, no_lab_coat: 0, no_mask: 0, no_goggles: 0 };
  const eventMap = new Map();
  const detections = [];
  const missingPpe = [];
  results.forEach((result, index) => {
    detections.push(...result.detections.map((item) => ({ ...item, frame: index + 1 })));
    Object.entries(result.risk.counts || {}).forEach(([key, value]) => { counts[key] = Math.max(counts[key] || 0, value); });
    (result.risk.events || []).forEach((event) => {
      const previous = eventMap.get(event.key);
      if (previous && previous.source !== event.source) {
        const sources = [...new Set([...(previous.evidence_sources || [previous.source]), ...(event.evidence_sources || [event.source])])];
        eventMap.set(event.key, {
          ...event,
          source: 'mixed_evidence',
          evidence_sources: sources,
          evidence_classes: [...new Set([...(previous.evidence_classes || []), ...(event.evidence_classes || [])])],
          label: event.key === 'no_mask' && sources.includes('behavior_inference') ? '未戴口罩（检测/行为推断）' : event.label
        });
      } else {
        eventMap.set(event.key, event);
      }
    });
    (result.risk.missing_ppe || []).forEach((item) => missingPpe.push({ ...item, frame: index + 1 }));
  });
  const events = [...eventMap.values()];
  const recommendations = events.length
    ? ['复核视频中出现风险的关键帧', '对进食、饮水和 PPE 风险进行现场确认', '保留视频分析结果作为巡检记录']
    : ['继续保持实验区域整洁和通风', '建议结合更多角度视频复核检测结果', '分析结果仅作为安全辅助判断'];
  const rawRisk = {
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
  const risk = applyVideoCoatConfirmation(rawRisk, videoCoatConfirmation(detections));
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
  const videoContext = { analysis_id: analysisContextId, coat_evidence: data.detections.filter(item => item.class_name === 'no_lab_coat') };
  latestAnalysis = data;
  let latestRecord = null;
  for (const frame of results) {
    const risk = applyVideoCoatConfirmation(frame.data.risk, data.risk.temporal_confirmation.no_lab_coat, [frame.frameIndex]);
    latestRecord = await saveHistory('video', frame.blob, { ...frame.data, risk, video_context: videoContext }, frame.frameIndex);
  }
  currentHistoryRecordId = latestRecord && latestRecord.id;
  reportBtn.disabled = false;
  $('#reportContent').classList.add('hidden');
  $('#reportState').textContent = '视频关键帧分析已就绪，点击按钮生成 DeepSeek 智能报告。';
  renderAnalysis(data, `视频分析完成 · ${data.frame_count} 个关键帧`);
}

function renderAnalysis(data, timeText, displayedRisk = data.risk) {
  updateAgentContext();
  const imageSize = data.image_size || {};
  if (imageSize.width && imageSize.height) setAdaptiveFrame($('#resultImageWrap'), imageSize.width, imageSize.height);
  $('#resultImage').onload = () => setAdaptiveFrame($('#resultImageWrap'), $('#resultImage').naturalWidth, $('#resultImage').naturalHeight);
  $('#resultImage').src = data.image;
  $('#resultImage').classList.remove('hidden');
  $('.result-placeholder').classList.add('hidden');
  $('#analysisTag').textContent = `${data.detections.length} 个目标`;
  $('#resultTime').textContent = timeText;
  renderRisk(displayedRisk);
  if (data.warning) $('#riskSummary').textContent += ` 模型提示：${data.warning}`;
  renderDetections(data.detections, data.risk);
  renderSignals(data.risk.counts, data.risk);
  if (currentMode !== 'camera') document.querySelector('#resultsSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function cameraSessionRisk() {
  const events = [...cameraConfirmedEvents.values()];
  return {
    events,
    summary: events.length
      ? `本次摄像头检测已确认 ${events.length} 类风险。`
      : '等待同一风险在最近三帧中至少检出两次。',
    recommendations: events.length ? ['查看「分析历史」中的摄像头截图，结合现场情况复核'] : []
  };
}

async function cameraTick() {
  const stream = cameraStream;
  if (!stream || cameraBusy || !cameraVideo.videoWidth) return;
  cameraBusy = true;
  const frameIndex = ++cameraFrameIndex;
  try {
    cameraCanvas.width = cameraVideo.videoWidth;
    cameraCanvas.height = cameraVideo.videoHeight;
    cameraCanvas.getContext('2d').drawImage(cameraVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);
    const blob = await new Promise((resolve) => cameraCanvas.toBlob(resolve, 'image/jpeg', 0.82));
    const data = await analyzeBlob(blob);
    if (cameraStream !== stream) return;
    const currentEvents = new Map((data.risk.events || []).map(event => [event.key, event]));
    cameraEventWindow = cameraEventWindow.filter(frame => frame.index >= frameIndex - 2);
    cameraEventWindow.push({ index: frameIndex, events: currentEvents });
    const confirmed = [...currentEvents.values()].filter(event =>
      cameraEventWindow.filter(frame => frame.events.has(event.key)).length >= 2
    );
    for (const event of confirmed) cameraConfirmedEvents.set(event.key, event);

    const confirmedKeys = new Set(confirmed.map(event => event.key));
    const frameRisk = {
      ...data.risk,
      events: confirmed,
      missing_ppe: (data.risk.missing_ppe || []).filter(item => confirmedKeys.has(item.class_name)),
      title: confirmed.length ? '摄像头风险记录' : '本帧无已确认风险',
      summary: confirmed.length ? `本帧复核确认 ${confirmed.length} 类风险。` : '本帧暂无已确认风险。',
      recommendations: confirmed.length ? ['结合检测截图和现场情况复核'] : []
    };
    latestAnalysis = { ...data, risk: frameRisk };
    renderAnalysis(latestAnalysis, `摄像头实时检测 · ${new Date().toLocaleTimeString()}`, cameraSessionRisk());
    reportBtn.disabled = true;
    $('#reportState').textContent = '摄像头实时检测不生成智能报告。';
    if (!confirmed.length) {
      lastHistoryKey = '';
      return;
    }
    const keys = [...confirmedKeys].sort().join('|');
    if (keys === lastHistoryKey && Date.now() - lastHistoryTime < 10000) return;
    const record = await saveHistory('camera', blob, latestAnalysis, frameIndex);
    if (cameraStream !== stream || !record) return;
    lastHistoryKey = keys;
    lastHistoryTime = Date.now();
    currentHistoryRecordId = record.id;
  } catch (error) {
    if (cameraStream === stream) {
      $('#analysisTag').textContent = 'CAMERA ERROR';
      $('#riskSummary').textContent = error.message;
    }
  } finally {
    if (cameraStream === stream) cameraBusy = false;
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
    $('#analysisTag').textContent = '实时检测';
    $('#cameraRunningStatus').classList.remove('hidden');
    cameraFrameIndex = 0;
    cameraEventWindow = [];
    cameraConfirmedEvents.clear();
    lastHistoryKey = '';
    renderRisk(cameraSessionRisk());
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
  $('#startCameraBtn').disabled = false;
  $('#stopCameraBtn').disabled = true;
  $('#cameraRunningStatus').classList.add('hidden');
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
  $('#analysisTag').textContent = '检测中';
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
  $('#historyCount').textContent = `${visibleItems.length} 条`;
  if (!visibleItems.length) {
    $('#historyGrid').innerHTML = '<div class="history-empty">暂无符合条件的历史分析记录</div>';
    return;
  }
  $('#historyGrid').innerHTML = visibleItems.map((item) => {
    const risk = item.risk || {};
    const eventLabels = (risk.events || []).map((event) => escapeHtml(event.label));
    const pendingLabels = (risk.pending_events || []).map(event => `待复核：${escapeHtml(event.label)}`);
    const events = [...eventLabels, ...pendingLabels].join('、') || '未发现明确风险';
    const frame = item.frame_index ? `<span>第 ${escapeHtml(item.frame_index)} 帧</span>` : '';
    const riskClass = (risk.events || []).length ? 'history-risk warning' : (pendingLabels.length ? 'history-risk pending' : 'history-risk safe');
    const selected = agentAttachmentIds.includes(item.id);
    return `<article class="history-card" data-preview-id="${escapeHtml(item.id)}"${selected ? ' aria-current="true"' : ''}>
      <button class="history-preview-btn" type="button" draggable="true" aria-label="放大查看${escapeHtml(item.source_label)}图片，${escapeHtml(formatHistoryTime(item.created_at))}"><img class="history-thumb" draggable="false" loading="lazy" src="${escapeHtml(item.image_url)}" alt="${escapeHtml(item.source_label)}截图"></button>
      <div class="history-card-body">
        <div class="history-card-top"><span class="history-source">${escapeHtml(item.source_label)}</span><div class="history-row-actions">
          <button class="history-attach" type="button" aria-label="加入对话" aria-pressed="${selected}"${agentBusy ? ' disabled' : ''}>${selected ? '已加入' : '加入'}</button>
          <button class="history-delete" data-history-id="${escapeHtml(item.id)}" type="button">删除</button>
        </div></div>
        <div class="history-time" title="${escapeHtml(formatHistoryTime(item.created_at))}">${escapeHtml(formatHistoryTime(item.created_at))} ${frame}</div>
        <p class="${riskClass}" title="${events}">${events}</p>
      </div>
    </article>`;
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
    historyLoaded = true;
    renderHistory();
    syncHistoryReferences();
    return true;
  } catch (error) {
    if (version !== historyLoadVersion) return false;
    $('#historyGrid').innerHTML = `<div class="history-empty">历史记录加载失败：${escapeHtml(error.message)}</div>`;
    if (historyPicker.open) renderHistoryPicker();
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
  syncHistoryReferences();
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
    if (!data.failed_count) {
      historyItems = [];
      historyLoaded = true;
      currentHistoryRecordId = null;
      syncHistoryReferences();
    }
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
  $('#historyFilters').querySelectorAll('.history-filter').forEach((item) => {
    item.classList.toggle('active', item === button);
    item.setAttribute('aria-pressed', String(item === button));
  });
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
  historyFullImage.removeAttribute('src');
  if (historyPreviewTrigger?.isConnected) historyPreviewTrigger.focus({ preventScroll: true });
  historyPreviewTrigger = null;
});

$('#historyGrid').addEventListener('click', async (event) => {
  const button = event.target.closest('.history-delete');
  if (historyClearing) return;
  if (!button) {
    const card = event.target.closest('.history-card');
    if (!card) return;
    if (event.target.closest('.history-attach')) {
      addAgentAttachment(card.dataset.previewId);
    } else {
      const preview = event.target.closest('.history-preview-btn');
      if (preview) openHistoryImage(card.dataset.previewId, preview);
    }
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

function renderSafetyReport(sections) {
  if (!sections) throw new Error('请重启本地服务后重新生成报告。');
  const content = $('#reportContent');
  content.innerHTML = `
    <article class="report-document" aria-labelledby="reportTitle">
      <h3 id="reportTitle">实验室安全分析报告</h3>
      <section class="report-summary">
        <h4>总体判断</h4>
        <p>${escapeHtml(sections.summary)}</p>
      </section>
      <section>
        <h4>主要发现</h4>
        <p>${escapeHtml(sections.findings)}</p>
      </section>
      <section>
        <h4>处置建议</h4>
        <ol>${sections.actions.map(action => `<li>${escapeHtml(action)}</li>`).join('')}</ol>
      </section>
    </article>`;
  content.classList.remove('hidden');
}

reportBtn.addEventListener('click', async () => {
  if (!latestAnalysis || currentMode === 'camera') return;
  const contextId = analysisContextId;
  reportBtn.disabled = true;
  reportBtn.querySelector('strong').textContent = '…';
  $('#reportState').textContent = '正在生成报告…';
  $('#reportContent').classList.add('hidden');
  try {
    const response = await fetch('/api/report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ detections: latestAnalysis.detections, risk: latestAnalysis.risk })
    });
    const data = await response.json();
    if (contextId !== analysisContextId) return;
    if (!response.ok) throw new Error(data.detail || '报告生成失败');
    renderSafetyReport(data.sections);
    $('#reportState').textContent = '报告已生成';
  } catch (error) {
    if (contextId === analysisContextId) $('#reportState').textContent = error.message;
  } finally {
    if (contextId === analysisContextId) {
      reportBtn.disabled = false;
      reportBtn.querySelector('strong').textContent = '↗';
    }
  }
});

function updateAgentContext() {
  const count = agentAttachmentIds.length;
  $('#agentContext').textContent = count ? `已选 ${count} / ${maxAgentAttachments} 张历史图片` :
    agentUseCurrentAnalysis && latestAnalysis ? '使用当前检测' :
      agentHasSavedAnalysis ? '沿用对话中的检测依据' : '自由对话';
  const useCurrent = $('#useCurrentAnalysis');
  useCurrent.classList.toggle('hidden', !latestAnalysis || (agentUseCurrentAnalysis && !count));
  useCurrent.disabled = agentBusy;
}

function historyRecords(ids) {
  return ids.map(id => historyItems.find(item => item.id === id) || { id });
}

function attachmentPreviewMarkup(record, index) {
  if (!record.image_url) {
    return `<span class="agent-attachment-missing">图 ${index} · ${historyLoaded ? '图片已删除' : '图片待加载'}</span>`;
  }
  const title = `图 ${index} · ${record.source_label}${record.frame_index ? `第 ${record.frame_index} 帧` : ''}`;
  return `<button class="agent-attachment-preview" type="button" data-preview="${escapeHtml(record.id)}" aria-label="查看${escapeHtml(title)}">
    <img src="${escapeHtml(record.image_url)}" alt=""><span>${escapeHtml(title)}</span>
  </button>`;
}

function attachmentMarkup(records, removable = false) {
  return records.map((record, index) => {
    const id = escapeHtml(record.id);
    return `<div class="agent-attachment" data-attachment-id="${id}" data-attachment-index="${index + 1}">
      ${attachmentPreviewMarkup(record, index + 1)}
      ${removable ? `<button class="agent-attachment-remove" type="button" data-remove="${id}" aria-label="移除图 ${index + 1}"${agentBusy ? ' disabled' : ''}>×</button>` : ''}
    </div>`;
  }).join('');
}

function selectedHistoryRecords() {
  return historyRecords(agentAttachmentIds);
}

function syncHistoryReferences() {
  if (historyLoaded) {
    const available = new Set(historyItems.map(item => item.id));
    agentAttachmentIds = agentAttachmentIds.filter(id => available.has(id));
    agentMessages.querySelectorAll('[data-attachment-id]').forEach(attachment => {
      const [record] = historyRecords([attachment.dataset.attachmentId]);
      if (!record.image_url || !attachment.querySelector('img')) {
        attachment.innerHTML = attachmentPreviewMarkup(record, attachment.dataset.attachmentIndex);
      }
    });
  }
  renderAgentAttachments();
  if (historyPicker.open) renderHistoryPicker();
}

function renderAgentAttachments() {
  const region = $('#agentAttachments');
  region.classList.toggle('hidden', !agentAttachmentIds.length);
  region.innerHTML = attachmentMarkup(selectedHistoryRecords(), true);
  $('#historyGrid').querySelectorAll('.history-attach').forEach(button => {
    const card = button.closest('.history-card');
    const selected = agentAttachmentIds.includes(card.dataset.previewId);
    if (selected) card.setAttribute('aria-current', 'true');
    else card.removeAttribute('aria-current');
    button.setAttribute('aria-pressed', String(selected));
    button.textContent = selected ? '已加入' : '加入';
    button.disabled = agentBusy;
  });
  updateAgentContext();
}

function addAgentAttachment(id) {
  if (agentBusy || !historyItems.some(item => item.id === id)) return;
  if (agentAttachmentIds.includes(id)) {
    $('#agentState').textContent = '这张图片已在对话中。';
    return;
  }
  if (agentAttachmentIds.length >= maxAgentAttachments) {
    $('#agentState').textContent = '一次最多加入 4 张图片，请先移除一张。';
    return;
  }
  agentAttachmentIds.push(id);
  renderAgentAttachments();
  $('#agentState').textContent = `已添加图 ${agentAttachmentIds.length}。`;
  if (appShell.dataset.workspace !== 'agent') window.location.hash = 'agent';
}

function removeAgentAttachments(ids) {
  agentAttachmentIds = agentAttachmentIds.filter(id => !ids.includes(id));
  renderAgentAttachments();
}

const agentPanel = $('#agentPanel');
const agentMessages = $('#agentMessages');
const agentEmpty = $('#agentEmpty');
$('#historyGrid').addEventListener('dragstart', event => {
  const preview = event.target.closest('.history-preview-btn');
  if (!preview || !event.dataTransfer) return;
  if (agentBusy || historyClearing) {
    event.preventDefault();
    return;
  }
  event.dataTransfer.effectAllowed = 'copy';
  event.dataTransfer.setData(historyDragType, preview.closest('.history-card').dataset.previewId);
});
$('#historyGrid').addEventListener('dragend', () => agentPanel.classList.remove('agent-dragging'));
agentPanel.addEventListener('dragover', event => {
  const types = Array.from(event.dataTransfer?.types || []);
  if (!types.includes(historyDragType) && !types.includes('Files')) return;
  event.preventDefault();
  const accepted = !agentBusy && types.includes(historyDragType);
  event.dataTransfer.dropEffect = accepted ? 'copy' : 'none';
  agentPanel.classList.toggle('agent-dragging', accepted);
});
agentPanel.addEventListener('dragleave', event => {
  if (!agentPanel.contains(event.relatedTarget)) agentPanel.classList.remove('agent-dragging');
});
agentPanel.addEventListener('drop', event => {
  const types = Array.from(event.dataTransfer?.types || []);
  if (!types.includes(historyDragType) && !types.includes('Files')) return;
  event.preventDefault();
  agentPanel.classList.remove('agent-dragging');
  if (types.includes(historyDragType)) addAgentAttachment(event.dataTransfer.getData(historyDragType));
  else $('#agentState').textContent = '请选择分析历史中的截图。';
});
agentPanel.addEventListener('click', event => {
  const preview = event.target.closest('.agent-attachment-preview');
  if (preview) {
    if (historyItems.some(item => item.id === preview.dataset.preview)) openHistoryImage(preview.dataset.preview, preview);
    else $('#agentState').textContent = '这条历史记录已删除，图片无法预览。';
  }
  const remove = event.target.closest('.agent-attachment-remove');
  if (remove && !agentBusy) {
    const index = agentAttachmentIds.indexOf(remove.dataset.remove);
    removeAgentAttachments([remove.dataset.remove]);
    const buttons = $('#agentAttachments').querySelectorAll('.agent-attachment-remove');
    (buttons[Math.min(index, buttons.length - 1)] || $('#agentInput')).focus();
    $('#agentState').textContent = '已移除图片。';
  }
});

function appendAgentMessage(role, content, records = [], source = '') {
  const message = document.createElement('div');
  message.className = `agent-message agent-message-${role}`;
  const label = document.createElement('b');
  label.textContent = role === 'user' ? '你' : '安全 Agent';
  const body = document.createElement('p');
  body.textContent = content;
  message.append(label, body);
  if (records.length) {
    const attachments = document.createElement('div');
    attachments.className = 'agent-message-attachments';
    attachments.innerHTML = attachmentMarkup(records);
    message.append(attachments);
  }
  if (source) {
    const note = document.createElement('small');
    note.className = 'agent-message-source';
    note.textContent = source;
    message.append(note);
  }
  agentEmpty.classList.add('hidden');
  agentMessages.appendChild(message);
  agentMessages.scrollTop = agentMessages.scrollHeight;
  return message;
}

function agentRequest(message, records) {
  const body = { message, session_id: agentSessionId };
  if (records.length) return { ...body, history_record_ids: records.map(record => record.id) };
  if (!agentUseCurrentAnalysis) return body;
  return {
    ...body,
    record_id: currentHistoryRecordId,
    analysis_id: analysisContextId,
    analysis: latestAnalysis ? { detections: latestAnalysis.detections, risk: latestAnalysis.risk,
      frame_count: latestAnalysis.frame_count, source_type: currentMode } : null
  };
}

async function sendAgentMessage(message, records) {
  const body = agentRequest(message, records);
  const key = JSON.stringify(body);
  if (agentRetry?.key !== key) agentRetry = { key, id: newLocalId() };
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...body, request_id: agentRetry.id })
  });
  const data = await response.json();
  if (!response.ok) {
    if (response.status === 404 && Array.isArray(data.detail?.missing_record_ids)) {
      const missing = data.detail.missing_record_ids;
      historyItems = historyItems.filter(item => !missing.includes(item.id));
      removeAgentAttachments(missing);
      renderHistory();
      syncHistoryReferences();
      loadHistory();
    }
    throw new Error(typeof data.detail === 'string' ? data.detail : data.detail?.message || 'Agent 请求失败');
  }
  agentRetry = null;
  agentSessionId = data.session_id;
  agentUseCurrentAnalysis = false;
  agentHasSavedAnalysis = data.has_analysis;
  rememberActiveChat();
  ++chatListVersion;
  chatItems = [data.conversation, ...chatItems.filter(item => item.id !== agentSessionId)];
  renderChatList();
  return data;
}

function rememberActiveChat() {
  try {
    if (agentSessionId) localStorage.setItem(activeChatStorageKey, agentSessionId);
    else localStorage.removeItem(activeChatStorageKey);
  } catch { /* Continue if storage is unavailable. */ }
}

function renderChatList() {
  if (!chatItems.length) {
    $('#chatList').innerHTML = '<p class="sidebar-message">暂无对话。发送消息后会自动保存。</p>';
    return;
  }
  $('#chatList').innerHTML = chatItems.map(chat => {
    const id = escapeHtml(chat.id);
    const title = escapeHtml(chat.title);
    const active = chat.id === agentSessionId;
    const disabled = agentBusy ? ' disabled' : '';
    return `<div class="chat-list-row"${active ? ' aria-current="true"' : ''}>
      <button type="button" class="chat-open" data-chat-id="${id}" title="${title}"${disabled}>${title}</button>
      <button type="button" class="text-btn chat-delete" data-delete-chat="${id}" aria-label="删除对话 ${title}"${disabled}>删除</button>
    </div>`;
  }).join('');
}

async function loadChatList() {
  const version = ++chatListVersion;
  try {
    const response = await fetch('/api/chats');
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '对话历史加载失败。');
    if (version !== chatListVersion) return;
    chatItems = data.items;
    $('#chatHistoryStatus').textContent = '';
    renderChatList();
  } catch (error) {
    if (version !== chatListVersion) return;
    $('#chatHistoryStatus').textContent = error.message || '对话历史加载失败，请刷新页面。';
  }
}

function resetAgentView(status = '') {
  agentSessionId = null;
  agentAttachmentIds = [];
  agentUseCurrentAnalysis = true;
  agentHasSavedAnalysis = false;
  agentRetry = null;
  rememberActiveChat();
  agentMessages.replaceChildren(agentEmpty);
  agentEmpty.classList.remove('hidden');
  $('#agentInput').value = '';
  $('#agentState').textContent = status;
  renderAgentAttachments();
  renderChatList();
}

function evidenceNote(mode) {
  if (mode === 'record_only') return '仅根据检测记录回答';
  if (mode === 'vision') return '依据所选截图和检测记录';
  return '';
}

async function openAgentChat(id, historyPromise = null) {
  if (agentBusy) return;
  const draft = id === agentSessionId ? $('#agentInput').value : '';
  setAgentBusy(true);
  try {
    const response = await fetch(`/api/chats/${encodeURIComponent(id)}`);
    const data = await response.json();
    if (!response.ok) {
      if (response.status === 404) {
        ++chatListVersion;
        chatItems = chatItems.filter(item => item.id !== id);
        if (agentSessionId === id) resetAgentView();
        else if (!agentSessionId) rememberActiveChat();
      }
      throw new Error(data.detail || '对话加载失败。');
    }
    if (historyPromise) await historyPromise;
    agentSessionId = data.id;
    agentRetry = null;
    agentUseCurrentAnalysis = false;
    agentHasSavedAnalysis = data.has_analysis;
    agentAttachmentIds = [];
    rememberActiveChat();
    agentMessages.replaceChildren(agentEmpty);
    agentEmpty.classList.remove('hidden');
    for (const turn of data.turns) {
      appendAgentMessage('user', turn.question, historyRecords(turn.history_record_ids));
      appendAgentMessage('bot', turn.answer, [], evidenceNote(turn.evidence_mode));
    }
    syncHistoryReferences();
    $('#agentInput').value = draft;
    $('#agentState').textContent = '';
    if (appShell.dataset.workspace === 'agent') {
      if (narrowWorkspace.matches) setHistorySidebar(false);
      $('#agentInput').focus({ preventScroll: true });
    }
  } catch (error) {
    $('#agentState').textContent = error.message || '对话加载失败。';
  } finally {
    setAgentBusy(false);
  }
}

$('#chatList').addEventListener('click', async event => {
  if (agentBusy) return;
  const remove = event.target.closest('[data-delete-chat]');
  const open = event.target.closest('[data-chat-id]');
  if (open) {
    await openAgentChat(open.dataset.chatId);
    return;
  }
  if (!remove) return;
  const id = remove.dataset.deleteChat;
  const chat = chatItems.find(item => item.id === id);
  if (!chat || !window.confirm(`删除对话「${chat.title}」？`)) return;
  setAgentBusy(true);
  try {
    const response = await fetch(`/api/chats/${encodeURIComponent(id)}`, { method: 'DELETE' });
    const data = await response.json();
    if (!response.ok && response.status !== 404) throw new Error(data.detail || '删除对话失败。');
    ++chatListVersion;
    chatItems = chatItems.filter(item => item.id !== id);
    if (agentSessionId === id) resetAgentView('对话已删除。');
    $('#chatHistoryStatus').textContent = '对话已删除。';
  } catch (error) {
    $('#chatHistoryStatus').textContent = error.message || '删除对话失败。';
  } finally {
    setAgentBusy(false);
  }
});

function setAgentBusy(busy) {
  agentBusy = busy;
  $('#agentInput').readOnly = busy;
  $('#agentSendBtn').disabled = busy;
  $('#clearChatBtn').disabled = busy;
  $('#showAgentHistory').disabled = busy;
  renderAgentAttachments();
  renderChatList();
}

$('#agentInput').addEventListener('keydown', (event) => {
  // Some IMEs report the confirmation key as Enter with keyCode 229.
  if (event.key !== 'Enter' || event.shiftKey || event.isComposing || event.keyCode === 229) return;
  event.preventDefault();
  if (!agentBusy && !event.repeat) event.currentTarget.form.requestSubmit();
});

$('#agentForm').addEventListener('submit', async (event) => {
  event.preventDefault();
  const input = $('#agentInput');
  const message = input.value.trim();
  if (!message || agentBusy) return;
  const records = selectedHistoryRecords();
  if (records.some(record => !record.image_url)) {
    $('#agentState').textContent = '请刷新检测记录，加载所选图片后再发送。';
    return;
  }
  input.value = '';
  setAgentBusy(true);
  const pending = appendAgentMessage('user', message, records);
  $('#agentState').textContent = records.length ? '正在复核所选图片…' : '正在回答…';
  try {
    const data = await sendAgentMessage(message, records);
    agentAttachmentIds = [];
    const source = evidenceNote(data.evidence_mode);
    appendAgentMessage('bot', data.answer, [], source);
    const focus = data.tool === 'search_history' ? null : data.discussion_focus;
    const focusParts = [...(focus?.frames || []).map((frame) => `第${frame}帧`), ...(focus?.categories || [])];
    $('#agentState').textContent = source || [agentToolLabels[data.tool], ...focusParts].filter(Boolean).join(' · ');
  } catch (error) {
    pending.remove();
    agentEmpty.classList.toggle('hidden', Boolean(agentMessages.querySelector('.agent-message')));
    input.value = message;
    $('#agentState').textContent = error.message || '请求失败，请稍后重试。';
  } finally {
    setAgentBusy(false);
    if (agentPanel.contains(document.activeElement)) input.focus({ preventScroll: true });
  }
});

$('#clearChatBtn').addEventListener('click', () => {
  if (agentBusy) return;
  resetAgentView('已开启新对话。');
  if (narrowWorkspace.matches) setHistorySidebar(false);
  $('#agentInput').focus({ preventScroll: true });
});

$('#useCurrentAnalysis').addEventListener('click', () => {
  if (agentBusy || !latestAnalysis) return;
  agentUseCurrentAnalysis = true;
  removeAgentAttachments(agentAttachmentIds);
  $('#agentState').textContent = '下一条消息将引用当前检测。';
  $('#agentInput').focus({ preventScroll: true });
});

const appShell = $('#top');
const workspaceMain = $('#workspaceMain');
const detectionView = $('#detectionView');
const workspaceSwitch = $('#workspaceSwitch');
const workspaceMenu = $('#workspaceMenu');
const sidebarToggle = $('#toggleHistory');
const sidebarBackdrop = $('#closeHistoryOverlay');
const narrowWorkspace = window.matchMedia('(max-width: 720px)');
let detectionHistoryExpanded = true;

function setWorkspaceMenu(open) {
  workspaceMenu.classList.toggle('hidden', !open);
  workspaceSwitch.setAttribute('aria-expanded', String(open));
}

function resizeDetectionFrames() {
  resizeAdaptiveFrame(previewImage.closest('.preview-frame'));
  resizeAdaptiveFrame($('#resultImageWrap'));
}

function setHistorySidebar(open) {
  appShell.classList.toggle('sidebar-closed', !open);
  sidebarToggle.setAttribute('aria-expanded', String(open));
  sidebarToggle.setAttribute('aria-label', open ? '收起历史侧栏' : '展开历史侧栏');
  sidebarBackdrop.classList.toggle('hidden', !open || !narrowWorkspace.matches);
  workspaceMain.inert = open && narrowWorkspace.matches;
  if (appShell.dataset.workspace === 'detection') requestAnimationFrame(resizeDetectionFrames);
}

function renderDetectionHistorySection() {
  const agent = appShell.dataset.workspace === 'agent';
  const expanded = !agent || detectionHistoryExpanded;
  const toggle = $('#toggleDetectionHistory');
  toggle.classList.toggle('hidden', !agent);
  toggle.setAttribute('aria-expanded', String(expanded));
  toggle.textContent = expanded ? '收起' : '展开';
  $('#detectionHistoryContent').classList.toggle('hidden', !expanded);
  toggle.closest('.detection-history').classList.toggle('is-collapsed', !expanded);
}

function showWorkspace() {
  const hash = window.location.hash;
  if (hash === '#workspaceMain') return;
  const agent = hash === '#agent' || hash === '#history';
  const workspace = agent ? 'agent' : 'detection';
  const label = agent ? 'Agent' : '检测';
  appShell.dataset.workspace = workspace;
  $('#workspaceLabel').textContent = label;
  workspaceSwitch.setAttribute('aria-label', `切换工作区，当前为${label}`);
  $('#shellTitle').textContent = agent ? '实验室安全助手' : '实验室安全检测';
  detectionView.classList.toggle('hidden', agent);
  $('#agent').classList.toggle('hidden', !agent);
  $('#clearChatBtn').classList.toggle('hidden', !agent);
  $('#chatHistory').classList.toggle('hidden', !agent);
  renderDetectionHistorySection();
  workspaceMenu.querySelectorAll('[data-workspace-link]').forEach(link => {
    if (link.dataset.workspaceLink === workspace) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  });
  setWorkspaceMenu(false);
  if (hash === '#history') setHistorySidebar(true);
  else if (narrowWorkspace.matches) setHistorySidebar(false);
  if (!agent) {
    requestAnimationFrame(() => {
      resizeDetectionFrames();
      if (hash === '#protocol' || hash === '#about') $(hash).scrollIntoView({ block: 'start' });
      else if (hash === '#analysis') detectionView.scrollTop = 0;
    });
  }
}

workspaceSwitch.addEventListener('click', () => {
  const open = workspaceMenu.classList.contains('hidden');
  setWorkspaceMenu(open);
  if (open) workspaceMenu.querySelector('[aria-current="page"]').focus();
});
workspaceMenu.addEventListener('click', event => {
  if (!event.target.closest('[data-workspace-link]')) return;
  setWorkspaceMenu(false);
  if (narrowWorkspace.matches) setHistorySidebar(false);
  workspaceSwitch.focus();
});
document.addEventListener('click', event => {
  if (!event.target.closest('.workspace-switch')) setWorkspaceMenu(false);
});
document.addEventListener('keydown', event => {
  if (event.key !== 'Escape' || event.defaultPrevented || document.querySelector('dialog[open]')) return;
  if (!workspaceMenu.classList.contains('hidden')) {
    setWorkspaceMenu(false);
    workspaceSwitch.focus();
  } else if (narrowWorkspace.matches && !appShell.classList.contains('sidebar-closed')) {
    setHistorySidebar(false);
    sidebarToggle.focus();
  }
});
sidebarToggle.addEventListener('click', () => setHistorySidebar(appShell.classList.contains('sidebar-closed')));
sidebarBackdrop.addEventListener('click', () => {
  setHistorySidebar(false);
  sidebarToggle.focus();
});
$('#showAgentHistory').addEventListener('click', () => {
  openHistoryPicker();
});
$('#toggleDetectionHistory').addEventListener('click', () => {
  detectionHistoryExpanded = !detectionHistoryExpanded;
  renderDetectionHistorySection();
});
narrowWorkspace.addEventListener('change', () => setHistorySidebar(!narrowWorkspace.matches));
window.addEventListener('hashchange', showWorkspace);

const historyPicker = $('#historyPicker');
const pickerGrid = $('#historyPickerGrid');
let pickerAttachmentIds = [];

function renderHistoryPicker() {
  const activeId = document.activeElement?.dataset.pickerId;
  const source = $('#pickerSource').value;
  if (historyLoaded) {
    const available = new Set(historyItems.map(item => item.id));
    pickerAttachmentIds = pickerAttachmentIds.filter(id => available.has(id));
  }
  const visible = historyItems.filter(item => source === 'all' || item.source === source);
  pickerGrid.innerHTML = visible.length ? visible.map(item => {
    const id = escapeHtml(item.id);
    const index = pickerAttachmentIds.indexOf(item.id) + 1;
    const label = escapeHtml(item.source_label);
    return `<article class="history-picker-card${index ? ' is-selected' : ''}">
      <button type="button" class="history-picker-thumb" data-picker-preview="${id}" aria-label="预览${label}截图">
        <img src="${escapeHtml(item.image_url)}" alt="" loading="lazy">
      </button>
      <div class="history-picker-card-body">
        <div class="history-time">${label}${item.frame_index ? ` · 第 ${escapeHtml(item.frame_index)} 帧` : ''}</div>
        <div class="history-time">${escapeHtml(formatHistoryTime(item.created_at))}</div>
        <button type="button" class="history-picker-option" data-picker-id="${id}" aria-pressed="${Boolean(index)}">${index ? `已选为图 ${index}` : '选择截图'}</button>
      </div>
    </article>`;
  }).join('') : `<p class="history-empty">${!historyLoaded ? '正在加载检测记录…' :
    historyItems.length ? '当前来源没有截图。' : '暂无截图。完成一次检测后，可在这里选择已保存的图片。'}</p>`;
  const count = pickerAttachmentIds.length;
  $('#historyPickerStatus').textContent = `已选 ${count} / ${maxAgentAttachments} 张`;
  const apply = $('#applyHistoryPicker');
  apply.disabled = !historyLoaded || (!count && !agentAttachmentIds.length);
  apply.textContent = count ? `加入 ${count} 张` : agentAttachmentIds.length ? '清除附件' : '加入对话';
  if (activeId) {
    Array.from(pickerGrid.querySelectorAll('[data-picker-id]')).find(button => button.dataset.pickerId === activeId)?.focus({ preventScroll: true });
  }
}

async function openHistoryPicker() {
  if (agentBusy || historyPicker.open) return;
  pickerAttachmentIds = [...agentAttachmentIds];
  $('#pickerSource').value = 'all';
  renderHistoryPicker();
  historyPicker.showModal();
  if (!historyLoaded && !(await loadHistory()) && historyPicker.open && !historyLoaded) {
    pickerGrid.innerHTML = '<p class="history-empty">检测记录加载失败。请关闭窗口后重试。</p>';
  }
}

$('#pickerSource').addEventListener('change', renderHistoryPicker);
pickerGrid.addEventListener('click', event => {
  const preview = event.target.closest('[data-picker-preview]');
  if (preview) {
    openHistoryImage(preview.dataset.pickerPreview, preview);
    return;
  }
  const id = event.target.closest('[data-picker-id]')?.dataset.pickerId;
  if (!id) return;
  if (pickerAttachmentIds.includes(id)) {
    pickerAttachmentIds = pickerAttachmentIds.filter(item => item !== id);
  } else if (pickerAttachmentIds.length < maxAgentAttachments) {
    pickerAttachmentIds.push(id);
  } else {
    $('#historyPickerStatus').textContent = '最多 4 张，请先取消一张。';
    return;
  }
  renderHistoryPicker();
});
$('#closeHistoryPicker').addEventListener('click', () => historyPicker.close());
$('#cancelHistoryPicker').addEventListener('click', () => historyPicker.close());
$('#applyHistoryPicker').addEventListener('click', () => {
  agentAttachmentIds = [...pickerAttachmentIds];
  renderAgentAttachments();
  historyPicker.close();
  $('#agentState').textContent = agentAttachmentIds.length ? '已更新所选截图。' : '已清除截图附件。';
  $('#agentInput').focus({ preventScroll: true });
});

setHistorySidebar(!narrowWorkspace.matches);
showWorkspace();
renderSignals();
loadModelInfo();
const initialHistory = loadHistory();
loadChatList();
try {
  const activeChat = localStorage.getItem(activeChatStorageKey);
  if (activeChat) openAgentChat(activeChat, initialHistory);
} catch { /* Continue if storage is unavailable. */ }
