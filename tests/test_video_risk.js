const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../app/frontend/app.js'), 'utf8');
const context = {};
vm.createContext(context);
vm.runInContext(source.slice(source.indexOf('function hasRisk('), source.indexOf('\nasync function analyzeVideo')), context);

function frame(negative, duplicates = 1) {
  const items = [{ class_name: negative ? 'no_lab_coat' : 'lab_coat', confidence: 0.8 }];
  if (negative) for (let i = 1; i < duplicates; i++) items.push({ ...items[0] });
  return { image: 'frame', detections: items, risk: {
    counts: { lab_coat: negative ? 0 : 1, no_lab_coat: negative ? duplicates : 0, no_mask: 1 },
    events: [{ key: 'no_mask', label: 'mask risk', source: 'visual_detection' }, ...(negative ? [{ key: 'no_lab_coat', label: 'coat risk', source: 'visual_detection' }] : [])],
    missing_ppe: negative ? [{ class_name: 'no_lab_coat', count: duplicates }] : []
  }};
}
const hasCoatRisk = data => data.risk.events.some(event => event.key === 'no_lab_coat');
const cases = [
  { sequence: [false, false, true, false, false, false, false, false], confirmed: false, pending: [3], confirmedFrames: [] },
  { sequence: [false, true, true, false], confirmed: true, pending: [], confirmedFrames: [2, 3] },
  { sequence: [true, false, true], confirmed: false, pending: [1, 3], confirmedFrames: [] },
  { sequence: [true, true], confirmed: true, pending: [], confirmedFrames: [1, 2] },
  { sequence: [true, true, false, true], confirmed: true, pending: [4], confirmedFrames: [1, 2] },
  { sequence: [false, false], confirmed: false, pending: [], confirmedFrames: [] }
];
for (const item of cases) {
  const frames = item.sequence.map(value => frame(value));
  const before = JSON.stringify(frames);
  const data = context.mergeVideoResults(frames);
  assert.equal(hasCoatRisk(data), item.confirmed);
  assert.deepEqual(Array.from(data.risk.temporal_confirmation.no_lab_coat.pending_frames), item.pending);
  assert.deepEqual(Array.from(data.risk.temporal_confirmation.no_lab_coat.confirmed_frames), item.confirmedFrames);
  assert.equal(data.detections.length, frames.length);
  assert.equal(data.risk.events.some(event => event.key === 'no_mask'), true);
  assert.equal(JSON.stringify(frames), before);
  assert.equal(data.risk.missing_ppe.filter(item => item.class_name === 'no_lab_coat').length, item.confirmedFrames.length);
}
const duplicated = context.mergeVideoResults([frame(false), frame(true, 2), frame(false)]);
assert.equal(hasCoatRisk(duplicated), false);
assert.equal(duplicated.risk.counts.no_lab_coat, 2); // raw model counts stay truthful
assert.equal(duplicated.risk.pending_events[0].key, 'no_lab_coat');
const historyRisk = context.applyVideoCoatConfirmation(frame(true).risk, { detected_frames: [2, 3], confirmed_frames: [2, 3], pending_frames: [] }, [2]);
assert.deepEqual(Array.from(historyRisk.temporal_confirmation.no_lab_coat.supporting_frames), [2, 3]);
const elements = {};
context.$ = selector => elements[selector] || (elements[selector] = { innerHTML: '', textContent: '' });
context.escapeHtml = value => String(value);
vm.runInContext(source.slice(source.indexOf('function renderRisk('), source.indexOf('\nfunction renderDetections(')), context);
context.renderRisk(duplicated.risk);
assert.equal((elements['#eventList'].innerHTML.match(/class="event-badge/g) || []).length, 1);
assert.equal(elements['#eventList'].innerHTML.includes('event-no_lab_coat'), false);
assert.equal(elements['#eventList'].innerHTML.includes('待复核'), true);
console.log('Video confirmation: 7 sequences, scoped history evidence and risk-panel rendering passed.');
