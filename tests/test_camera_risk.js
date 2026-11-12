const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const frontend = path.join(__dirname, '../app/frontend');
const source = fs.readFileSync(path.join(frontend, 'app.js'), 'utf8');
const html = fs.readFileSync(path.join(frontend, 'index.html'), 'utf8');

async function testCameraHistory() {
  let now = 100_000;
  let events = [];
  const saved = [];
  let failSave = false;
  let releaseAnalysis;
  const video = { videoWidth: 0, videoHeight: 480, play: async () => {} };
  const elements = new Map();
  class Clock extends Date {
    static now() { return now; }
  }
  const context = vm.createContext({
    $: selector => {
      if (!elements.has(selector)) elements.set(selector, { classList: { add() {}, remove() {} } });
      return elements.get(selector);
    },
    Date: Clock,
    escapeHtml: String,
    cameraVideo: video,
    cameraCanvas: {
      getContext: () => ({ drawImage() {} }),
      toBlob: resolve => resolve('frame')
    },
    reportBtn: {},
    navigator: { mediaDevices: { getUserMedia: async () => ({ getTracks: () => [{ stop() {} }] }) } },
    resetReportState() {},
    setInterval: () => 1,
    clearInterval() {},
    analyzeBlob: async () => {
      const data = { risk: { events, missing_ppe: events.map(event => ({ class_name: event.key })) } };
      if (releaseAnalysis) await releaseAnalysis;
      return data;
    },
    renderAnalysis(data, timeText, displayedRisk) {
      context.rendered = data;
      context.renderRisk(displayedRisk);
    },
    async saveHistory(source, blob, data, frameIndex) {
      if (failSave) throw new Error('save failed');
      saved.push({ source, blob, data, frameIndex });
      return { id: `record-${saved.length}` };
    }
  });
  vm.runInContext(source.slice(source.indexOf('let cameraStream'), source.indexOf('let historyItems')), context);
  vm.runInContext('let latestAnalysis, currentHistoryRecordId;', context);
  vm.runInContext(source.slice(source.indexOf('function renderRisk('), source.indexOf('function renderDetections(')), context);
  vm.runInContext(source.slice(source.indexOf('function cameraSessionRisk('), source.indexOf("analyzeBtn.addEventListener('click'")), context);
  await context.startCamera();
  video.videoWidth = 640;

  const labels = () => [...elements.get('#eventList').innerHTML.matchAll(/event-badge event-(\w+)/g)].map(match => match[1]);
  const tick = async (keys, elapsed = 1000) => {
    events = keys.map(key => ({ key, label: key }));
    now += elapsed;
    await context.cameraTick();
  };
  await tick(['no_gloves', 'no_gloves']);
  assert.deepEqual(labels(), [], 'two boxes in one frame do not confirm a risk');
  await tick(['no_head_mask']);
  assert.deepEqual(labels(), [], 'different risks in adjacent frames do not confirm each other');
  assert.equal(saved.length, 0);
  await tick(['no_gloves', 'no_goggles']);
  assert.deepEqual(labels(), ['no_gloves'], 'two hits within three frames confirm a risk');
  assert.equal(saved.length, 1);
  assert.equal(saved[0].source, 'camera');
  assert.equal(saved[0].blob, 'frame');
  assert.equal(saved[0].frameIndex, 3);
  assert.equal(saved[0].data, context.rendered);
  assert.deepEqual(Array.from(saved[0].data.risk.events, event => event.key), ['no_gloves']);
  assert.deepEqual(Array.from(saved[0].data.risk.missing_ppe, item => item.class_name), ['no_gloves']);
  await tick(['no_gloves']);
  assert.equal(saved.length, 1, 'suppress an unchanged risk within ten seconds');
  await tick(['no_gloves'], 9000);
  assert.equal(saved.length, 2);
  await tick(['no_head_mask']);
  assert.equal(saved.length, 2, 'a hit outside the three-frame window cannot confirm a risk');
  await tick([]);
  await tick(['no_head_mask']);
  assert.equal(saved.length, 3);
  assert.deepEqual(labels(), ['no_gloves', 'no_head_mask']);
  assert.deepEqual(Array.from(saved[2].data.risk.events, event => event.key), ['no_head_mask'], 'history events are supported by the saved frame');
  for (let frame = 0; frame < 3; frame += 1) await tick([]);
  assert.deepEqual(labels(), ['no_gloves', 'no_head_mask'], 'confirmed labels survive an empty window');
  assert.equal(saved.length, 3, 'retained labels alone do not generate history');

  await tick(['no_goggles']);
  failSave = true;
  await tick(['no_goggles']);
  assert.deepEqual(labels(), ['no_gloves', 'no_head_mask', 'no_goggles']);
  failSave = false;
  await tick(['no_goggles']);
  assert.equal(saved.length, 4, 'failed saves are eligible for retry');

  await tick(['no_mask']);
  let release;
  releaseAnalysis = new Promise(resolve => { release = resolve; });
  const oldFrame = tick(['no_mask']);
  await Promise.resolve();
  context.stopCamera();
  assert.deepEqual(labels(), ['no_gloves', 'no_head_mask', 'no_goggles'], 'stopping leaves confirmed labels visible');
  video.videoWidth = 0;
  await context.startCamera();
  assert.deepEqual(labels(), [], 'starting a new session clears the previous labels');
  video.videoWidth = 640;
  release();
  await oldFrame;
  assert.equal(saved.length, 4, 'a stopped stream cannot save a stale inference');
  await tick(['no_mask']);
  assert.deepEqual(labels(), [], 'hits from the previous session do not confirm a risk');
  await tick(['no_mask']);
  assert.deepEqual(labels(), ['no_mask']);
  assert.equal(saved[4].frameIndex, 2, 'a new camera session restarts frame numbering');
  assert.doesNotMatch(html, /id="liveAlert"/);
  assert.doesNotMatch(source, /alertFromRisk/);
}

testCameraHistory().then(() => console.log('Camera risk records: two-of-three confirmation, persistent labels, history, retry and session reset passed.'), error => {
  console.error(error);
  process.exitCode = 1;
});
