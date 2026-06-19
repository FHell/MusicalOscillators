// Musical Oscillators - Piano UI
// Interfaces with the WASM module to synthesize Stuart-Landau oscillator audio.

import init, { synthesize, synthesize_with_matrix, note_frequency } from './pkg/musical_oscillators_wasm.js';

// ── Constants ──────────────────────────────────────────────────────────────

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const MATRIX_SIZE = 12;
const PITCHES_C4_B4 = Array.from({ length: 12 }, (_, i) => 60 + i);

function noteName(pitch) {
  return NOTE_NAMES[pitch % 12] + Math.floor(pitch / 12 - 1);
}

function isBlack(pitch) {
  const pc = pitch % 12;
  return [1, 3, 6, 8, 10].includes(pc);
}

// ── State ──────────────────────────────────────────────────────────────────

let wasmReady = false;
let audioCtx = null;

// Selected piano key (MIDI pitch)
let selectedPitch = 60;

// Notes to synthesize: [{pitch, velocity, start, duration}]
let notes = [];
let noteTime = 0.0; // accumulates start times

// 12×12 coupling matrix (flat, row-major)
let couplingMatrix = new Float64Array(MATRIX_SIZE * MATRIX_SIZE);
let matrixDragging = false;
let matrixDragValue = 1; // value being set during drag

// ── DOM refs ──────────────────────────────────────────────────────────────

const statusEl = document.getElementById('status');
const noteListEl = document.getElementById('note-list');
const pianoEl = document.getElementById('piano-keys');
const matrixGridEl = document.getElementById('matrix-grid');
const matrixColLabelsEl = document.getElementById('matrix-col-labels');
const matrixRowsEl = document.getElementById('matrix-rows');

const muSlider = document.getElementById('mu');
const muVal = document.getElementById('mu-val');
const kappaSlider = document.getElementById('kappa');
const kappaVal = document.getElementById('kappa-val');
const attackSlider = document.getElementById('attack');
const attackVal = document.getElementById('attack-val');
const releaseSlider = document.getElementById('release');
const releaseVal = document.getElementById('release-val');

// ── Status helpers ─────────────────────────────────────────────────────────

function setStatus(msg, type = '') {
  statusEl.textContent = msg;
  statusEl.className = type;
}

// ── Piano rendering ────────────────────────────────────────────────────────

function buildPiano() {
  pianoEl.innerHTML = '';
  const whiteKeys = PITCHES_C4_B4.filter(p => !isBlack(p));
  const whiteCount = whiteKeys.length;
  const whiteWidth = 43; // px including margin

  PITCHES_C4_B4.forEach((pitch, idx) => {
    const key = document.createElement('div');
    key.classList.add('key');
    const pc = pitch % 12;
    const black = isBlack(pitch);
    key.classList.add(black ? 'black' : 'white');
    key.dataset.pitch = pitch;

    const name = NOTE_NAMES[pc];
    key.title = noteName(pitch);

    if (!black) {
      key.textContent = name;
    }

    if (black) {
      // Position black keys absolutely between white keys
      // Compute position based on chromatic index
      const offsets = { 1: 0.65, 3: 1.65, 6: 3.65, 8: 4.65, 10: 5.65 };
      const left = offsets[pc] * whiteWidth + 8;
      key.style.left = left + 'px';
    }

    key.addEventListener('mousedown', (e) => {
      e.preventDefault();
      selectKey(pitch);
    });

    pianoEl.appendChild(key);
  });

  // Mark initial selected
  updatePianoHighlight();
}

function selectKey(pitch) {
  selectedPitch = pitch;
  updatePianoHighlight();
}

function updatePianoHighlight() {
  document.querySelectorAll('.key').forEach(el => {
    const p = parseInt(el.dataset.pitch);
    el.classList.toggle('active', p === selectedPitch);
  });
}

// ── Note sequencer ─────────────────────────────────────────────────────────

function addNote() {
  const duration = 0.4;
  notes.push({ pitch: selectedPitch, velocity: 100, start: noteTime, duration });
  noteTime += duration * 0.9; // slight overlap
  renderNoteList();
}

function clearNotes() {
  notes = [];
  noteTime = 0.0;
  renderNoteList();
}

function removeNote(idx) {
  notes.splice(idx, 1);
  // Recalculate start times to be sequential
  let t = 0;
  notes.forEach(n => {
    n.start = t;
    t += n.duration * 0.9;
  });
  noteTime = t;
  renderNoteList();
}

function renderNoteList() {
  noteListEl.innerHTML = '';
  if (notes.length === 0) {
    const li = document.createElement('li');
    li.style.color = '#444';
    li.textContent = 'No notes added yet.';
    noteListEl.appendChild(li);
    return;
  }
  notes.forEach((n, idx) => {
    const li = document.createElement('li');
    li.innerHTML = `<span>${noteName(n.pitch)} vel=${n.velocity} @${n.start.toFixed(2)}s dur=${n.duration.toFixed(2)}s</span>`;
    const btn = document.createElement('button');
    btn.className = 'remove-btn';
    btn.textContent = '✕';
    btn.title = 'Remove note';
    btn.addEventListener('click', () => removeNote(idx));
    li.appendChild(btn);
    noteListEl.appendChild(li);
  });
}

// ── Matrix rendering ───────────────────────────────────────────────────────

function buildMatrix() {
  // Column labels
  matrixColLabelsEl.innerHTML = '';
  PITCHES_C4_B4.forEach(p => {
    const span = document.createElement('span');
    span.textContent = NOTE_NAMES[p % 12];
    matrixColLabelsEl.appendChild(span);
  });

  // Rows
  matrixRowsEl.innerHTML = '';
  for (let i = 0; i < MATRIX_SIZE; i++) {
    const rowEl = document.createElement('div');
    rowEl.className = 'matrix-row';

    const rowLabel = document.createElement('div');
    rowLabel.className = 'matrix-row-label';
    rowLabel.textContent = NOTE_NAMES[PITCHES_C4_B4[i] % 12];
    rowEl.appendChild(rowLabel);

    const cells = document.createElement('div');
    cells.className = 'matrix-row-cells';

    for (let j = 0; j < MATRIX_SIZE; j++) {
      const cell = document.createElement('div');
      cell.className = 'matrix-cell';
      cell.dataset.i = i;
      cell.dataset.j = j;
      cell.title = `K[${NOTE_NAMES[PITCHES_C4_B4[i] % 12]},${NOTE_NAMES[PITCHES_C4_B4[j] % 12]}]`;

      cell.addEventListener('mousedown', (e) => {
        e.preventDefault();
        matrixDragging = true;
        const curVal = couplingMatrix[i * MATRIX_SIZE + j];
        matrixDragValue = curVal > 0 ? 0 : 1;
        setMatrixCell(i, j, matrixDragValue, cell);
      });

      cell.addEventListener('mouseenter', (e) => {
        if (matrixDragging) {
          setMatrixCell(i, j, matrixDragValue, cell);
        }
      });

      cells.appendChild(cell);
    }
    rowEl.appendChild(cells);
    matrixRowsEl.appendChild(rowEl);
  }

  document.addEventListener('mouseup', () => { matrixDragging = false; });
}

function setMatrixCell(i, j, value, cellEl) {
  couplingMatrix[i * MATRIX_SIZE + j] = value;
  if (!cellEl) {
    cellEl = matrixRowsEl.querySelector(`[data-i="${i}"][data-j="${j}"]`);
  }
  if (cellEl) {
    cellEl.dataset.value = value;
  }
}

function shuffleMatrix() {
  for (let i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
    couplingMatrix[i] = Math.random() > 0.5 ? 1 : 0;
  }
  refreshMatrixDisplay();
}

function clearMatrix() {
  couplingMatrix.fill(0);
  refreshMatrixDisplay();
}

function refreshMatrixDisplay() {
  matrixRowsEl.querySelectorAll('.matrix-cell').forEach(cell => {
    const i = parseInt(cell.dataset.i);
    const j = parseInt(cell.dataset.j);
    cell.dataset.value = couplingMatrix[i * MATRIX_SIZE + j];
  });
}

function matrixIsNonTrivial() {
  return couplingMatrix.some(v => v !== 0);
}

// ── Audio playback ─────────────────────────────────────────────────────────

function ensureAudioContext() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (audioCtx.state === 'suspended') {
    audioCtx.resume();
  }
  return audioCtx;
}

async function synthesizeAndPlay() {
  if (!wasmReady) {
    setStatus('WASM not loaded yet.', 'error');
    return;
  }
  if (notes.length === 0) {
    setStatus('Add some notes first.', 'error');
    return;
  }

  setStatus('Synthesizing...', 'info');

  const pitchArr = new Uint8Array(notes.map(n => n.pitch));
  const velArr = new Uint8Array(notes.map(n => n.velocity));
  const startArr = new Float64Array(notes.map(n => n.start));
  const durArr = new Float64Array(notes.map(n => n.duration));

  const samplerate = 44100;
  const mu = parseFloat(muSlider.value);
  const kappa = parseFloat(kappaSlider.value);
  const attack = parseFloat(attackSlider.value);
  const rel = parseFloat(releaseSlider.value);
  const tail = 0.3;
  const gain = 0.9;
  const damp = 5.0;

  let samples;
  try {
    if (matrixIsNonTrivial()) {
      const matrixF64 = new Float64Array(couplingMatrix);
      samples = synthesize_with_matrix(
        pitchArr, velArr, startArr, durArr,
        samplerate, mu, matrixF64, kappa,
        damp, attack, rel, tail, gain
      );
    } else {
      samples = synthesize(
        pitchArr, velArr, startArr, durArr,
        samplerate, mu, kappa, damp, attack, rel, tail, gain
      );
    }
  } catch (e) {
    setStatus('Synthesis error: ' + e.message, 'error');
    return;
  }

  if (!samples || samples.length === 0) {
    setStatus('No audio generated.', 'error');
    return;
  }

  const ctx = ensureAudioContext();
  const buffer = ctx.createBuffer(1, samples.length, samplerate);
  buffer.copyToChannel(samples, 0);

  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  source.start();

  setStatus(`Playing ${(samples.length / samplerate).toFixed(2)}s of audio (${notes.length} notes).`, 'success');
}

// ── Slider wiring ──────────────────────────────────────────────────────────

function wireSlider(slider, display, decimals = 2) {
  display.textContent = parseFloat(slider.value).toFixed(decimals);
  slider.addEventListener('input', () => {
    display.textContent = parseFloat(slider.value).toFixed(decimals);
  });
}

// ── Init ───────────────────────────────────────────────────────────────────

async function main() {
  setStatus('Loading WASM...', 'info');

  try {
    await init();
    wasmReady = true;
    setStatus('WASM loaded. Ready.', 'success');
  } catch (e) {
    setStatus('Failed to load WASM: ' + e.message, 'error');
    return;
  }

  buildPiano();
  buildMatrix();
  renderNoteList();

  wireSlider(muSlider, muVal, 0);
  wireSlider(kappaSlider, kappaVal, 3);
  wireSlider(attackSlider, attackVal, 3);
  wireSlider(releaseSlider, releaseVal, 3);

  document.getElementById('btn-add-note').addEventListener('click', addNote);
  document.getElementById('btn-clear-notes').addEventListener('click', clearNotes);
  document.getElementById('btn-synthesize').addEventListener('click', synthesizeAndPlay);
  document.getElementById('btn-matrix-shuffle').addEventListener('click', shuffleMatrix);
  document.getElementById('btn-matrix-clear').addEventListener('click', clearMatrix);
}

main();
