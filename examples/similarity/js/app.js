// The region similarity demo: which regions of two images look alike to a model. The model describes
// every cell of its last feature map with a feature vector (../js/inference.js); the similarity of two
// cells, one in each image, is the dot product of their vectors. The lines connect the most similar
// pairs of cells:
//   a cell under the pointer (or keyboard focus)   its strongest connections to the other image
//   selected cells                                 the strongest connections that start at them
//   otherwise                                      the strongest connections between the images

const SIDES = ["Image A", "Image B"];
const lower = (side) => SIDES[side].replace("Image", "image");   // "image A", inside a sentence
const START_IMAGES = ["Dog & cat", "Dog & cat indoors"];
const MAX_CANVAS = 1024;   // the image canvas never needs more pixels than it is shown with

const $ = (id) => document.getElementById(id);
const els = {
  pair: $("pair"),
  links: $("links"),
  status: $("status"),
  modelSelect: $("model-select"),
  countInput: $("count-input"),
  clearButton: $("clear-button"),
};

const state = {
  model: null,          // the selected entry of MODELS
  panels: [],           // per image: its elements, source bitmap and model output
  similarity: null,     // { values: cells of A x cells of B, rows, cols } or null
  pointer: null,        // { side, cell } under the pointer
  focus: null,          // { side, cell } with keyboard focus
  count: 30,
};

// ── panels ────────────────────────────────────────────────────────────────────

function createPanel(side) {
  const root = $("panel-template").content.firstElementChild.cloneNode(true);
  const q = (selector) => root.querySelector(selector);
  const panel = {
    side, root,
    name: q(".image-name"),
    frame: q(".frame"),
    canvas: q("canvas"),
    grid: q(".grid"),
    busy: q(".busy"),
    busyText: q(".busy-text"),
    picker: q(".picker"),
    fileInput: q(".file-input"),
    cells: [],
    selected: new Set(),
    source: null,       // ImageBitmap
    result: null,       // the model's output for this image, with the model's id
  };
  const pickerId = `picker-${side}`;
  q(".side").textContent = SIDES[side];
  root.setAttribute("aria-label", SIDES[side]);
  panel.grid.setAttribute("aria-label", `Regions of ${lower(side)}`);
  panel.picker.id = pickerId;
  panel.picker.setAttribute("aria-label", `Choose ${lower(side)}`);
  q(".picker h3").textContent = `Choose ${lower(side)}`;
  const choose = q(".choose");
  choose.setAttribute("popovertarget", pickerId);
  choose.setAttribute("aria-label", `Change ${lower(side)}`);
  q(".close").onclick = () => panel.picker.hidePopover();

  // the example images, then an upload button
  q(".thumbs").replaceChildren(...SAMPLES.map((sample) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "thumb";
    const image = Object.assign(document.createElement("img"), { src: sample.url, alt: "", loading: "lazy" });
    const label = Object.assign(document.createElement("span"), { textContent: sample.label });
    button.append(image, label);
    button.onclick = async () => {
      panel.picker.hidePopover();
      await setImage(panel, await fetch(sample.url).then((r) => r.blob()), sample.label);
    };
    return button;
  }));
  panel.picker.addEventListener("toggle", (e) => {
    if (e.newState === "open") panel.picker.querySelector(".thumb").focus();
  });
  q(".upload").onclick = () => panel.fileInput.click();
  panel.fileInput.onchange = async () => {
    const file = panel.fileInput.files[0];
    panel.fileInput.value = "";
    panel.picker.hidePopover();
    if (file) await setImage(panel, file, file.name);
  };

  // dropping an image file onto the panel
  root.addEventListener("dragover", (e) => {
    e.preventDefault();
    root.classList.add("dragging");
  });
  root.addEventListener("dragleave", (e) => {
    if (!root.contains(e.relatedTarget)) root.classList.remove("dragging");
  });
  root.addEventListener("drop", async (e) => {
    e.preventDefault();
    root.classList.remove("dragging");
    const file = [...e.dataTransfer.files].find((f) => f.type.startsWith("image/"));
    if (file) await setImage(panel, file, file.name);
  });

  panel.grid.addEventListener("pointerleave", () => setPointer(null));
  panel.grid.addEventListener("keydown", (e) => moveFocus(panel, e));
  els.pair.insertBefore(root, els.links);
  return panel;
}

// one button per cell of the feature map; only one cell per grid is in the tab order (arrow keys move)
function buildGrid(panel, height, width) {
  if (panel.cells.length === height * width && panel.width === width) return;
  panel.width = width;
  panel.selected.clear();
  panel.grid.style.gridTemplateColumns = `repeat(${width}, 1fr)`;
  panel.cells = Array.from({ length: height * width }, (_, cell) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "cell";
    button.tabIndex = cell === 0 ? 0 : -1;
    button.setAttribute("aria-label", `${SIDES[panel.side]}, row ${Math.floor(cell / width) + 1}, column ${cell % width + 1}`);
    button.setAttribute("aria-pressed", "false");
    button.onclick = () => toggleCell(panel, cell);
    button.onpointerenter = () => setPointer({ side: panel.side, cell });
    button.onfocus = () => {
      panel.cells.forEach((c, i) => (c.tabIndex = i === cell ? 0 : -1));
      state.focus = button.matches(":focus-visible") ? { side: panel.side, cell } : null;
      draw();
    };
    button.onblur = () => {
      state.focus = null;
      draw();
    };
    return button;
  });
  panel.grid.replaceChildren(...panel.cells);
}

function moveFocus(panel, e) {
  const steps = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -panel.width, ArrowDown: panel.width };
  if (!(e.key in steps)) return;
  e.preventDefault();
  const from = panel.cells.indexOf(document.activeElement);
  const to = from + steps[e.key];
  const sameRow = Math.floor(to / panel.width) === Math.floor(from / panel.width);
  if (to >= 0 && to < panel.cells.length && (Math.abs(steps[e.key]) > 1 || sameRow)) panel.cells[to].focus();
}

function toggleCell(panel, cell) {
  panel.selected.has(cell) ? panel.selected.delete(cell) : panel.selected.add(cell);
  panel.cells[cell].setAttribute("aria-pressed", panel.selected.has(cell));
  draw();
}

function clearSelection() {
  for (const panel of state.panels) {
    panel.selected.clear();
    panel.cells.forEach((c) => c.setAttribute("aria-pressed", "false"));
  }
  draw();
}

function setPointer(target) {
  state.pointer = target;
  draw();
}

function showBusy(panel, text) {
  panel.busy.hidden = !text;
  panel.busyText.textContent = text || "";
}

// ── images and the model ──────────────────────────────────────────────────────

// the model runs one request at a time, in order: loading a model, then the images
let queue = Promise.resolve();
const enqueue = (task) => (queue = queue.then(task, task));

async function setImage(panel, blob, name) {
  let bitmap;
  try {
    bitmap = await createImageBitmap(blob);
  } catch {
    setStatus(`Could not open ${name} as an image.`);
    return;
  }
  panel.source?.close();
  panel.source = bitmap;
  panel.name.textContent = `· ${name}`;
  // the centred square of the image
  const side = Math.min(bitmap.width, bitmap.height);
  panel.canvas.width = panel.canvas.height = Math.min(MAX_CANVAS, side);
  panel.canvas.getContext("2d").drawImage(bitmap, (bitmap.width - side) / 2, (bitmap.height - side) / 2, side, side,
    0, 0, panel.canvas.width, panel.canvas.height);
  panel.selected.clear();
  panel.cells.forEach((c) => c.setAttribute("aria-pressed", "false"));
  panel.result = null;
  updateSimilarity();
  await analyze(panel);
}

function analyze(panel) {
  return enqueue(async () => {
    const model = state.model;
    if (!model || !panel.source) return;
    showBusy(panel, "Computing features…");
    try {
      const bitmap = await createImageBitmap(panel.canvas, { resizeWidth: model.size, resizeHeight: model.size, resizeQuality: "high" });
      const result = await runModel(bitmap);
      if (state.model !== model) return;          // another model was selected meanwhile
      panel.result = { ...result, model: model.id };
      buildGrid(panel, panel.result.height, panel.result.width);
      updateSimilarity();
    } catch (err) {
      console.error(err);
      setStatus(`The model failed on ${lower(panel.side)}: ${err.message}`);
    } finally {
      showBusy(panel, "");
    }
  });
}

async function selectModel(entry) {
  els.modelSelect.disabled = true;
  state.model = null;
  for (const panel of state.panels) panel.result = null;
  updateSimilarity();
  await enqueue(async () => {
    const show = (text) => state.panels.forEach((panel) => showBusy(panel, text));
    try {
      const { backend } = await loadModel(entry, show);
      state.model = entry;
      console.info(`${entry.label} runs on the ${backend.name === "webgpu" ? "GPU (WebGPU)" : "CPU (WebAssembly)"}: ${backend.reason}`);
    } catch (err) {
      console.error(err);
      setStatus(`Could not load ${entry.label}: ${err.message}`);
    } finally {
      show("");
    }
  });
  els.modelSelect.disabled = false;
  await Promise.all(state.panels.map(analyze));
}

// ── similarity and connections ────────────────────────────────────────────────

// the dot product of every cell of image A with every cell of image B
function updateSimilarity() {
  const [a, b] = state.panels.map((p) => p.result);
  state.similarity = null;
  if (a && b && a.model === b.model) {
    const n = a.height * a.width, m = b.height * b.width;
    const values = new Float32Array(n * m);
    for (let c = 0; c < a.channels; c++) {
      const fa = a.features.subarray(c * n, (c + 1) * n);
      const fb = b.features.subarray(c * m, (c + 1) * m);
      for (let i = 0; i < n; i++) {
        const x = fa[i];
        if (x === 0) continue;
        const row = i * m;
        for (let j = 0; j < m; j++) values[row + j] += x * fb[j];
      }
    }
    state.similarity = { values, rows: n, cols: m };
    els.countInput.max = n * m;
  }
  draw();
}

// the connections of the current view (hover, selection or all pairs): all of them, strongest first,
// as [similarity, cell of A, cell of B]; the first state.count are drawn
function connections() {
  const sim = state.similarity;
  if (!sim) return [];
  const [a, b] = state.panels;
  const target = state.pointer ?? state.focus;
  let keep = () => true;
  if (target) keep = (i, j) => (target.side === 0 ? i === target.cell : j === target.cell);
  else if (a.selected.size || b.selected.size) keep = (i, j) => a.selected.has(i) || b.selected.has(j);
  const pairs = [];
  for (let i = 0; i < sim.rows; i++) {
    for (let j = 0; j < sim.cols; j++) if (keep(i, j)) pairs.push([sim.values[i * sim.cols + j], i, j]);
  }
  return pairs.sort((x, y) => y[0] - x[0]);
}

const SVG = "http://www.w3.org/2000/svg";

function svg(name, attributes) {
  const el = document.createElementNS(SVG, name);
  for (const [k, v] of Object.entries(attributes)) el.setAttribute(k, v);
  return el;
}

// the centre of a cell, in the coordinates of the pair container
function cellCenter(panel, cell, origin) {
  const r = panel.frame.getBoundingClientRect();
  const width = panel.width, height = panel.cells.length / width;
  return [
    r.left - origin.left + ((cell % width) + 0.5) * (r.width / width),
    r.top - origin.top + (Math.floor(cell / width) + 0.5) * (r.height / height),
  ];
}

function draw() {
  const all = connections();
  const pairs = all.slice(0, state.count);
  const [a, b] = state.panels;
  const target = state.pointer ?? state.focus;
  for (const panel of state.panels) {
    panel.cells.forEach((c, i) => c.classList.toggle("target", target?.side === panel.side && target.cell === i));
  }
  els.clearButton.disabled = !a.selected.size && !b.selected.size;

  const origin = els.pair.getBoundingClientRect();
  els.links.setAttribute("viewBox", `0 0 ${origin.width} ${origin.height}`);
  // side by side: the curves leave and enter horizontally; stacked (narrow screens): vertically
  const horizontal = b.frame.getBoundingClientRect().left >= a.frame.getBoundingClientRect().right - 1;
  // a line's strength is relative to all connections of the current view, from the weakest (faint)
  // to the strongest (full), whether they are drawn or not
  const strongest = all[0]?.[0], weakest = all.at(-1)?.[0];
  const lines = [], dots = [];
  for (const [value, i, j] of [...pairs].reverse()) {     // the strongest on top
    const t = strongest > weakest ? (value - weakest) / (strongest - weakest) : 1;
    const [x1, y1] = cellCenter(a, i, origin);
    const [x2, y2] = cellCenter(b, j, origin);
    const d = horizontal
      ? `M${x1},${y1} C${(x1 + x2) / 2},${y1} ${(x1 + x2) / 2},${y2} ${x2},${y2}`
      : `M${x1},${y1} C${x1},${(y1 + y2) / 2} ${x2},${(y1 + y2) / 2} ${x2},${y2}`;
    const width = 1 + 3 * t;
    lines.push(svg("path", { d, class: "halo", "stroke-width": width + 2, opacity: 0.1 + 0.5 * t }));
    lines.push(svg("path", { d, class: "line", "stroke-width": width, opacity: 0.15 + 0.85 * t }));
    for (const [x, y] of [[x1, y1], [x2, y2]]) dots.push(svg("circle", { cx: x, cy: y, r: 2.5 + 2 * t, class: "dot", opacity: 0.3 + 0.7 * t }));
  }
  els.links.replaceChildren(...lines, ...dots);
  describe(pairs, target);
}

function setStatus(text) {
  els.status.textContent = text;
}

const where = (side, cell) => {
  const width = state.panels[side].width;
  return `${lower(side)}, row ${Math.floor(cell / width) + 1}, column ${cell % width + 1}`;
};

// the connections in words, for everyone and for screen readers
function describe(pairs, target) {
  if (!state.similarity) return setStatus(state.model ? "Computing…" : "");
  if (!pairs.length) return setStatus("No connections.");
  const [, i, j] = pairs[0];
  const best = target ? where(1 - target.side, target.side === 0 ? j : i) : `${where(0, i)} and ${where(1, j)}`;
  const [a, b] = state.panels;
  const selected = a.selected.size + b.selected.size;
  const scope = target ? `from ${where(target.side, target.cell)}`
    : selected ? `from the ${selected} selected region${selected > 1 ? "s" : ""}`
    : `of ${(state.similarity.rows * state.similarity.cols).toLocaleString("en")} pairs`;
  setStatus(`${pairs.length} strongest connection${pairs.length > 1 ? "s" : ""} ${scope}. Most similar: ${best}.`);
}

// ── start ─────────────────────────────────────────────────────────────────────

function init() {
  state.panels = SIDES.map((_, side) => createPanel(side));
  fillModelMenu(els.modelSelect);
  els.modelSelect.onchange = () => selectModel(MODELS.find((m) => m.id === els.modelSelect.value));
  els.countInput.oninput = () => {
    const n = Math.round(Number(els.countInput.value));
    if (n >= 1) {
      state.count = Math.min(n, Number(els.countInput.max));
      draw();
    }
  };
  els.clearButton.onclick = clearSelection;
  new ResizeObserver(() => draw()).observe(els.pair);

  selectModel(MODELS[0]);
  state.panels.forEach(async (panel, side) => {
    const sample = SAMPLES.find((s) => s.label === START_IMAGES[side]);
    await setImage(panel, await fetch(sample.url).then((r) => r.blob()), sample.label);
  });
}

init();
