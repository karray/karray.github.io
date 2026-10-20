const TOP_N = 14;
const CLASS_MODES = ["imagenet_classes", "groupmap_bbs"];   // modes that need a classifier
const MAX_CANVAS = 1024;   // the image canvas never needs more pixels than it is shown with

const $ = (id) => document.getElementById(id);
const els = {
  main: $("main-section"),
  imgCanvas: $("img-canvas"),
  heatmapCanvas: $("heatmap-canvas"),
  boxCanvas: $("bounding-boxes-canvas"),
  grid: $("grid"),
  loading: $("loading-indicator"),
  loadingText: $("loading-text"),
  modelSelect: $("model-select"),
  modelNote: $("model-note"),
  modeSelect: $("mode-select"),
  paletteSelect: $("palette-select"),
  opacity: $("heatmap-opacity"),
  presets: $("predefined-files"),
  upload: $("upload-input"),
  startButton: $("start-button"),
  cameraButton: $("switch-camera-button"),
  predictions: $("prediction-list"),
  clearSelection: $("clear-selection"),
  tracking: $("tracking-controls"),
  objectList: $("tracking-objects-list"),
};

const state = {
  model: null,            // the selected entry of MODELS
  ready: false,           // the model is loaded
  busy: false,            // the model is running
  again: false,           // run once more when the current run ends
  classNames: [],
  palettes: {},
  grouper: null,
  source: null,           // an ImageBitmap, or the video element (video file or camera)
  playing: false,
  result: null,           // the last output of the model
  selectedCells: new Set(),
  selectedClasses: new Set(),
  objects: [],            // tracked objects: { id, name, color, cells, embedding, visible, pos }
  activeObject: null,
  tracking: { threshold: 0.5, ema: 0.75, resolution: 224 },
  stream: null,
  cameras: [],
  cameraIndex: 0,
};

const video = document.createElement("video");
video.muted = true;
video.playsInline = true;

// ── loading ───────────────────────────────────────────────────────────────────

const jsonCache = {};
const loadJson = (url) => (jsonCache[url] ??= fetch(url).then((r) => r.json()));

function showLoading(text) {
  els.loading.style.display = text ? "" : "none";
  els.loadingText.textContent = text || "";
}

async function selectModel(entry) {
  state.ready = false;
  els.main.classList.remove("ready");
  els.modelSelect.disabled = true;
  try {
    const { backend, classNames } = await loadModel(entry, showLoading);
    log(backend.name === "webgpu"
      ? `${entry.label} runs on the GPU (WebGPU: ${backend.reason})`
      : `${entry.label} runs on the CPU (WebAssembly): ${backend.reason}`);

    state.model = entry;
    state.classNames = classNames;
    for (const option of els.modeSelect.options) {
      option.disabled = (!entry.classes && CLASS_MODES.includes(option.value)) ||
        (option.value === "groupmap_bbs" && entry.classes !== IMAGENET_CLASSES);
    }
    if (els.modeSelect.selectedOptions[0].disabled) els.modeSelect.value = "predict";
    state.result = null;
    state.objects = [];          // object features belong to the previous model
    state.activeObject = null;
    renderObjectList();
    clearSelections();
    showModelNote(entry);
    await matchExample(entry);
    state.ready = true;
    els.main.classList.add("ready");
    showLoading("");
    analyze();
  } catch (err) {
    showLoading(`Failed to load ${entry.label}`);
    log(err);
  } finally {
    els.modelSelect.disabled = false;
  }
}

function showModelNote(entry) {
  const notes = [];
  if (entry.scale) notes.push(`Expects ${entry.scale}.`);
  if (entry.pooling) {
    notes.push(`The prediction is the model's own (${entry.pooling === "attention" ? "attention pooling" : "class token"}); ` +
      "each region is classified on its own, so the mean of the regions is not the prediction.");
  }
  els.modelNote.textContent = notes.join(" ");
  els.modelNote.hidden = !notes.length;
}

async function matchExample(entry) {
  const shown = SAMPLES.find((s) => s.url === els.presets.value);
  const wanted = entry.group === "pathology" ? "pathology" : "photo";
  if (!shown || sampleGroup(shown) === wanted) return;
  els.presets.value = SAMPLES.find((s) => sampleGroup(s) === wanted).url;
  await els.presets.onchange();
}

// ── sources: image, video file, camera ────────────────────────────────────────

function sourceSize(source) {
  return source === video ? [video.videoWidth, video.videoHeight] : [source.width, source.height];
}

function setSource(source) {
  if (state.source && state.source !== video && state.source !== source) state.source.close();
  state.source = source;
  const side = Math.min(MAX_CANVAS, ...sourceSize(source));
  for (const canvas of [els.imgCanvas, els.boxCanvas]) {
    canvas.width = side;
    canvas.height = side;
  }
  clearSelections();
  for (const obj of state.objects) obj.pos = null;
  els.startButton.disabled = source !== video;
  analyze();
}

function stopCamera() {
  if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
  state.stream = null;
  video.srcObject = null;
}

async function showImage(blob) {
  setPlaying(false);
  stopCamera();
  video.removeAttribute("src");
  setSource(await createImageBitmap(blob));
}

async function showVideo(url) {
  setPlaying(false);
  stopCamera();
  if (video.src.startsWith("blob:")) URL.revokeObjectURL(video.src);
  video.src = url;
  await new Promise((resolve) => video.addEventListener("loadeddata", resolve, { once: true }));
  setSource(video);
}

// The first press asks for permission to use the camera; later presses switch to the next camera.
async function nextCamera() {
  const constraints = { height: { ideal: 1024 } };
  if (state.cameras.length > 1) {
    state.cameraIndex = (state.cameraIndex + 1) % state.cameras.length;
    constraints.deviceId = { exact: state.cameras[state.cameraIndex].deviceId };
  }
  setPlaying(false);
  stopCamera();
  video.removeAttribute("src");
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: constraints });
  } catch (err) {
    log("Camera not available: " + err);
    return;
  }
  if (!state.cameras.length) {
    state.cameras = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
    const current = state.stream.getVideoTracks()[0].getSettings().deviceId;
    state.cameraIndex = Math.max(0, state.cameras.findIndex((c) => c.deviceId === current));
  }
  video.srcObject = state.stream;
  await video.play();
  els.presets.value = "";
  setSource(video);
  setPlaying(true);
}

function setPlaying(on) {
  state.playing = on && state.source === video;
  els.main.classList.toggle("paused", !state.playing);
  if (state.playing) {
    clearSelections();
    video.play();
    analyze();
  } else {
    video.pause();
  }
}

// ── running the model ─────────────────────────────────────────────────────────

// the centred square of the source, shown on the image canvas
function drawSource() {
  const [w, h] = sourceSize(state.source);
  const side = Math.min(w, h);
  const canvas = els.imgCanvas;
  canvas.getContext("2d").drawImage(state.source, (w - side) / 2, (h - side) / 2, side, side, 0, 0, canvas.width, canvas.height);
}

async function analyze() {
  if (!state.ready || !state.source) return;
  if (state.busy) {
    state.again = true;
    return;
  }
  state.busy = true;
  try {
    drawSource();
    const size = state.model.size;
    const bitmap = await createImageBitmap(els.imgCanvas, { resizeWidth: size, resizeHeight: size, resizeQuality: "high" });
    state.result = await runModel(bitmap);
    render();
  } catch (err) {
    log(err);
  } finally {
    state.busy = false;
  }
  if (state.again || state.playing) {
    state.again = false;
    requestAnimationFrame(analyze);
  }
}

// ── computations on a result ──────────────────────────────────────────────────
// features: channels x cells, cellLogits: classes x cells (cells in row-major order)

const cellCount = (r) => r.height * r.width;
const allCells = (r) => [...Array(cellCount(r)).keys()];

// the exact readout of a region: the mean of its cells' logits (all cells: the model's logits);
function regionLogits(r, cells = null) {
  if (!r.cellLogits) return null;
  if (!cells && r.logits) return r.logits;
  cells ??= allCells(r);
  const n = cellCount(r);
  const logits = new Float32Array(r.classes);
  for (let k = 0; k < r.classes; k++) {
    let sum = 0;
    for (const i of cells) sum += r.cellLogits[k * n + i];
    logits[k] = sum / cells.length;
  }
  return logits;
}

// LaFAM: the mean of the ReLU features of each cell
function lafamMap(r) {
  const n = cellCount(r);
  const map = new Float32Array(n);
  for (let c = 0; c < r.channels; c++) {
    for (let i = 0; i < n; i++) map[i] += Math.max(0, r.features[c * n + i]);
  }
  return map;
}

// CAM: the sum of the selected classes' cell logits (the bias shifts every cell alike)
function camMap(r, classes) {
  const n = cellCount(r);
  const map = new Float32Array(n);
  for (const k of classes) {
    for (let i = 0; i < n; i++) map[i] += r.cellLogits[k * n + i];
  }
  return map;
}

// the top class of every cell
function cellClasses(r) {
  const n = cellCount(r);
  const classIds = new Int32Array(n);
  const logits = new Float32Array(n).fill(-Infinity);
  for (let k = 0; k < r.classes; k++) {
    for (let i = 0; i < n; i++) {
      const v = r.cellLogits[k * n + i];
      if (v > logits[i]) {
        logits[i] = v;
        classIds[i] = k;
      }
    }
  }
  return { classIds, logits };
}

// the mean features of the given cells
function embedding(r, cells) {
  const n = cellCount(r);
  const e = new Float32Array(r.channels);
  for (let c = 0; c < r.channels; c++) {
    for (const i of cells) e[c] += r.features[c * n + i] / cells.length;
  }
  return e;
}

// the cosine similarity of every cell's features to an embedding
function cosineMap(r, e) {
  const n = cellCount(r);
  const dot = new Float32Array(n), norm = new Float32Array(n);
  let eNorm = 0;
  for (let c = 0; c < r.channels; c++) {
    eNorm += e[c] * e[c];
    for (let i = 0; i < n; i++) {
      const f = r.features[c * n + i];
      dot[i] += f * e[c];
      norm[i] += f * f;
    }
  }
  return dot.map((d, i) => d / (Math.sqrt(norm[i] * eNorm) + 1e-12));
}

function minMax(values) {
  let lo = Infinity, hi = -Infinity;
  for (const v of values) {
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  return Float32Array.from(values, (v) => (v - lo) / (hi - lo + 1e-6));
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}

function topN(logits, n, threshold = -Infinity) {
  return [...logits.keys()]
    .sort((a, b) => logits[b] - logits[a])
    .slice(0, n)
    .filter((i) => logits[i] >= threshold);
}

function upsampleBilinear(src, srcW, srcH, dstW, dstH) {
  const dst = new Float32Array(dstW * dstH);
  for (let y = 0; y < dstH; y++) {
    const gy = (y + 0.5) * (srcH / dstH) - 0.5;
    const y0 = Math.max(0, Math.min(srcH - 1, Math.floor(gy))), y1 = Math.min(srcH - 1, y0 + 1), wy = gy - y0;
    for (let x = 0; x < dstW; x++) {
      const gx = (x + 0.5) * (srcW / dstW) - 0.5;
      const x0 = Math.max(0, Math.min(srcW - 1, Math.floor(gx))), x1 = Math.min(srcW - 1, x0 + 1), wx = gx - x0;
      dst[y * dstW + x] =
        (1 - wy) * ((1 - wx) * src[y0 * srcW + x0] + wx * src[y0 * srcW + x1]) +
        wy * ((1 - wx) * src[y1 * srcW + x0] + wx * src[y1 * srcW + x1]);
    }
  }
  return dst;
}

// ── drawing ───────────────────────────────────────────────────────────────────

// one pixel per cell; the canvas is scaled up by CSS without smoothing
function drawMap(r, values, colorOf) {
  const canvas = els.heatmapCanvas;
  canvas.width = r.width;
  canvas.height = r.height;
  const image = new ImageData(r.width, r.height);
  values.forEach((v, i) => {
    image.data.set(colorOf(v), 4 * i);
    image.data[4 * i + 3] = 255;
  });
  canvas.getContext("2d").putImageData(image, 0, 0);
}

function paletteColor(v) {
  const palette = state.palettes[els.paletteSelect.value];
  return palette[Math.round(v * (palette.length - 1))];
}

function hueColor(v) {   // for maps of class or group ids
  const h = (20 + 300 * v) % 360;
  const f = (n) => {
    const k = (n + h / 30) % 12;
    return Math.round(255 * (0.5 - 0.5 * Math.max(-1, Math.min(k - 3, 9 - k, 1))));
  };
  return [f(0), f(8), f(4)];
}

function setupGrid(r) {
  if (els.grid.children.length !== cellCount(r)) {
    els.grid.replaceChildren(...allCells(r).map((i) => {
      const div = document.createElement("div");
      div.dataset.idx = i;
      return div;
    }));
  }
  els.grid.style.gridTemplateColumns = `repeat(${r.width}, 1fr)`;
  for (const div of els.grid.children) {
    div.innerHTML = "";
    div.className = "";
    div.style.borderColor = div.style.boxShadow = "";
  }
}

function markCell(i, color = null) {
  const div = els.grid.children[i];
  div.classList.add("selected");
  if (color) {
    div.style.borderColor = color;
    div.style.boxShadow = `0 0 20px ${color} inset`;
  }
}

function labelCells(values, ids) {
  [...els.grid.children].forEach((div, i) => {
    div.className = "class-display";
    div.innerHTML = `${values[i].toFixed(2)}<br/>${ids[i]}`;
  });
}

function showPredictions(logits, threshold) {
  if (!logits) {
    showList([`${state.model.label} has no classifier. The heatmap shows where its features respond most (LaFAM).`]);
    return;
  }
  const probs = softmax(logits);
  els.predictions.replaceChildren(
    ...topN(logits, TOP_N, threshold).map((k) => {
      const div = document.createElement("div");
      div.className = "prediction" + (state.selectedClasses.has(k) ? " selected" : "");
      div.dataset.idx = k;
      div.innerHTML = `<label><b>${state.classNames[k]}</b> (Logits: ${logits[k].toFixed(2)}; Softmax: ${Math.round(probs[k] * 100)}%)</label>`;
      div.append(Object.assign(document.createElement("progress"), { value: probs[k], max: 1 }));
      return div;
    })
  );
}

function showList(lines) {
  els.predictions.replaceChildren(
    ...lines.map((text) => Object.assign(document.createElement("div"), { className: "list-line", innerText: text }))
  );
}

// ── views ─────────────────────────────────────────────────────────────────────

const views = {
  // LaFAM, or CAM of the selected classes; predictions of the image or of the selected cells
  predict(r) {
    const classes = [...state.selectedClasses];
    drawMap(r, minMax(classes.length ? camMap(r, classes) : lafamMap(r)), paletteColor);
    const cells = [...state.selectedCells];
    cells.forEach((i) => markCell(i));
    showPredictions(cells.length ? regionLogits(r, cells) : regionLogits(r), cells.length || r.classes <= 20 ? -Infinity : 1.7);
  },

  // the top class of every cell
  imagenet_classes(r) {
    const { classIds, logits } = cellClasses(r);
    labelCells(logits, classIds);
    drawMap(r, minMax(classIds), hueColor);
    showList([...new Set(classIds)].sort((a, b) => a - b).map((k) => `${k}: ${state.classNames[k]}`));
  },

  // cells grouped into object categories, with boxes around connected cells of one group
  groupmap_bbs(r) {
    const { classIds, logits } = cellClasses(r);
    const groups = Array.from(classIds, (k) => state.grouper.of(k));
    labelCells(logits, groups);
    drawMap(r, minMax(groups), hueColor);
    const rows = Array.from({ length: r.height }, (_, y) => groups.slice(y * r.width, (y + 1) * r.width));
    drawBoundingBoxes(findBoundingBoxes(findAreas(rows, [-1])), els.boxCanvas, r.width, state.grouper.name);
    showList([...new Set(groups)].filter((g) => g >= 0).sort((a, b) => a - b).map((g) => `${g}: ${state.grouper.name(g)}`));
  },

  // every object at the centre of mass of the cells similar to it
  object_tracking(r) {
    const active = state.objects.find((o) => o.id === state.activeObject);
    if (active) active.cells.forEach((i) => markCell(i, active.color));
    drawMap(r, minMax(active?.embedding ? cosineMap(r, active.embedding) : lafamMap(r)), paletteColor);
    showPredictions(active?.cells.length ? regionLogits(r, active.cells) : regionLogits(r), active?.cells.length ? -Infinity : 1.7);

    const { threshold, ema, resolution: res } = state.tracking;
    const ctx = els.boxCanvas.getContext("2d");
    for (const obj of state.objects) {
      if (!obj.embedding || !obj.visible) continue;
      const map = upsampleBilinear(cosineMap(r, obj.embedding), r.width, r.height, res, res);
      let w = 0, sx = 0, sy = 0;
      map.forEach((v, i) => {
        if (v < threshold) return;
        w += v;
        sx += v * ((i % res) + 0.5);
        sy += v * (Math.floor(i / res) + 0.5);
      });
      if (w === 0) {
        obj.pos = null;
        continue;
      }
      let pos = { x: sx / w / res, y: sy / w / res };   // as a fraction of the image side
      if (obj.pos) pos = { x: ema * obj.pos.x + (1 - ema) * pos.x, y: ema * obj.pos.y + (1 - ema) * pos.y };
      obj.pos = pos;

      const x = pos.x * els.boxCanvas.width, y = pos.y * els.boxCanvas.height, s = els.boxCanvas.width * 0.05;
      ctx.strokeStyle = ctx.fillStyle = obj.color;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(x - s, y);
      ctx.lineTo(x + s, y);
      ctx.moveTo(x, y - s);
      ctx.lineTo(x, y + s);
      ctx.stroke();
      ctx.font = "16px sans-serif";
      ctx.fillText(obj.name, x + 5, y - 5);
    }
  },
};

const SELECTABLE = { predict: true, object_tracking: true };   // modes in which grid cells can be selected

function render() {
  const r = state.result;
  if (!r) return;
  const mode = els.modeSelect.value;
  setupGrid(r);
  els.grid.classList.toggle("selectable", !!SELECTABLE[mode]);
  els.paletteSelect.disabled = !SELECTABLE[mode];
  els.tracking.classList.toggle("hidden", mode !== "object_tracking");
  els.boxCanvas.getContext("2d").clearRect(0, 0, els.boxCanvas.width, els.boxCanvas.height);
  els.clearSelection.disabled = !state.selectedCells.size && !state.selectedClasses.size;
  views[mode](r);
}

function clearSelections() {
  state.selectedCells.clear();
  state.selectedClasses.clear();
}

// ── tracked objects ───────────────────────────────────────────────────────────

const OBJECT_COLORS = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF", "#FF00FF"];

function addObject() {
  const obj = {
    id: Date.now(),
    name: `Object ${state.objects.length + 1}`,
    color: OBJECT_COLORS[state.objects.length % OBJECT_COLORS.length],
    cells: [],
    embedding: null,
    visible: true,
    pos: null,
  };
  state.objects.push(obj);
  state.activeObject = obj.id;
  renderObjectList();
  render();
}

function removeObject(id) {
  state.objects = state.objects.filter((o) => o.id !== id);
  if (state.activeObject === id) state.activeObject = state.objects[0]?.id ?? null;
  renderObjectList();
  render();
}

// an object is the mean features of its cells; its name is the top class of these cells
function toggleObjectCell(obj, i) {
  obj.cells = obj.cells.includes(i) ? obj.cells.filter((c) => c !== i) : [...obj.cells, i];
  const r = state.result;
  if (obj.cells.length) {
    obj.embedding = embedding(r, obj.cells);
    const logits = regionLogits(r, obj.cells);
    if (logits) {
      const k = topN(logits, 1)[0];
      obj.name = `[${k}] ${state.classNames[k]}`;
    }
  } else {
    obj.embedding = null;
  }
  obj.pos = null;
  renderObjectList();
}

function renderObjectList() {
  els.objectList.replaceChildren(
    ...state.objects.map((obj) => {
      const div = document.createElement("div");
      div.className = "prediction" + (obj.id === state.activeObject ? " selected" : "");
      div.style.borderLeft = `5px solid ${obj.color}`;
      const name = Object.assign(document.createElement("span"), { innerText: obj.name });
      name.style.flex = "1";
      name.onclick = () => {
        state.activeObject = obj.id;
        renderObjectList();
        render();
      };
      const remove = Object.assign(document.createElement("button"), { innerText: "X", className: "remove-object" });
      remove.onclick = () => removeObject(obj.id);
      div.append(name, remove);
      return div;
    })
  );
}

// ── events ────────────────────────────────────────────────────────────────────

els.modelSelect.onchange = () => selectModel(MODELS.find((m) => m.id === els.modelSelect.value));
els.modeSelect.onchange = () => {
  clearSelections();
  render();
};
els.paletteSelect.onchange = render;
els.opacity.oninput = () => (els.heatmapCanvas.style.opacity = els.opacity.value);
els.startButton.onclick = () => setPlaying(!state.playing);
els.cameraButton.onclick = nextCamera;
$("upload-button").onclick = () => els.upload.click();
$("add-object-btn").onclick = addObject;

els.presets.onchange = async () => {
  const url = els.presets.value;
  if (url.endsWith(".mp4")) return showVideo(url);
  showImage(await (await fetch(url)).blob());
};

els.upload.onchange = () => {
  const file = els.upload.files[0];
  if (!file) return;
  els.presets.value = "";
  if (file.type.startsWith("video/")) showVideo(URL.createObjectURL(file));
  else showImage(file);
};

els.grid.onclick = (e) => {
  const mode = els.modeSelect.value;
  const div = e.target.closest("#grid > div");
  if (!div || !SELECTABLE[mode] || state.playing) return;
  const i = Number(div.dataset.idx);
  if (mode === "object_tracking") {
    const obj = state.objects.find((o) => o.id === state.activeObject);
    if (obj) toggleObjectCell(obj, i);
  } else {
    state.selectedClasses.clear();
    state.selectedCells.has(i) ? state.selectedCells.delete(i) : state.selectedCells.add(i);
  }
  render();
};

els.predictions.onclick = (e) => {
  const div = e.target.closest(".prediction");
  if (!div || els.modeSelect.value !== "predict" || state.playing) return;
  const k = Number(div.dataset.idx);
  state.selectedClasses.has(k) ? state.selectedClasses.delete(k) : state.selectedClasses.add(k);
  render();
};

els.clearSelection.onclick = () => {
  clearSelections();
  render();
};

video.onended = () => setPlaying(false);

for (const [id, key, digits] of [["tracking-threshold", "threshold", 2], ["tracking-ema", "ema", 2]]) {
  $(id).oninput = (e) => {
    state.tracking[key] = parseFloat(e.target.value);
    $(`${id}-val`).innerText = state.tracking[key].toFixed(digits);
  };
}
$("tracking-resolution").onchange = (e) => (state.tracking.resolution = parseInt(e.target.value));

// ── debugging ─────────────────────────────────────────────────────────────────

function log(msg) {
  console.warn(msg);
  const area = $("log");
  area.textContent += msg + "\n";
  area.scrollTop = area.scrollHeight;
}
window.addEventListener("error", (e) => log(e.message));
window.addEventListener("unhandledrejection", (e) => log(e.reason));

// clears the caches and the service worker, then reloads (the "Hard reset" button)
async function updateServiceWorker() {
  if ("caches" in window) await Promise.all((await caches.keys()).map((key) => caches.delete(key)));
  if ("serviceWorker" in navigator) {
    for (const registration of await navigator.serviceWorker.getRegistrations()) await registration.unregister();
  }
  localStorage.removeItem("lafam_tutorial_shown");
  window.location.reload();
}

// ── start ─────────────────────────────────────────────────────────────────────

function makeGrouper(include, exclude) {
  const names = Object.keys(include);
  const groupOf = {};
  names.forEach((name, g) => include[name].forEach((k) => (groupOf[k] = g)));
  const excluded = new Set(Object.values(exclude).flat());
  return {
    of: (k) => (excluded.has(k) ? -1 : groupOf[k] ?? -1),
    name: (g) => (g < 0 ? "---" : names[g]),
  };
}

async function init() {
  // the camera is only offered here; permission is asked on the first press of the camera button
  const devices = (await navigator.mediaDevices?.enumerateDevices?.()) ?? [];
  if (devices.some((d) => d.kind === "videoinput")) els.main.classList.add("camera-available");

  const [palettes, include, exclude] = await Promise.all(
    ["palettes", "include_groups", "exclude_groups"].map((name) => loadJson(`assets/data/${name}.json`))
  );
  state.palettes = palettes;
  state.grouper = makeGrouper(include, exclude);
  els.paletteSelect.replaceChildren(...Object.keys(palettes).map((p) => new Option(p, p)));
  fillModelMenu(els.modelSelect);

  els.presets.querySelector('option[value$=".mp4"]').before(
    ...SAMPLES.map((s) => new Option(`${sampleGroup(s) === "pathology" ? "🔬" : "🖼️"} ${s.label}`, s.url)));
  els.presets.selectedIndex = 1;          // start with the first example image
  els.presets.onchange();
  await selectModel(MODELS[0]);
}

init();
