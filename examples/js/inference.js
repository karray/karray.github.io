// Runs the models of examples/models/models.js for any demo, in a worker (worker.js), one model at
// a time. A model is downloaded when it is selected; the service worker caches the downloads for the
// whole site, so selecting it again reads it from the cache.
//
//   const { backend, classNames } = await loadModel(MODELS[0], showStatus);
//   const { features, cellLogits, channels, classes, height, width } = await runModel(bitmap);
//   fillModelMenu(select);   // the models as <option>s, grouped by how they were trained

const inferenceWorker = new Worker("/examples/js/worker.js", { type: "module" });
const pendingCalls = new Map();
let nextCallId = 0;

inferenceWorker.onmessage = ({ data }) => {
  const { resolve, reject } = pendingCalls.get(data.id);
  pendingCalls.delete(data.id);
  data.error ? reject(new Error(data.error)) : resolve(data);
};

function callWorker(message, transfer = []) {
  const id = nextCallId++;
  return new Promise((resolve, reject) => {
    pendingCalls.set(id, { resolve, reject });
    inferenceWorker.postMessage({ ...message, id }, transfer);
  });
}

async function download(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  const total = Number(response.headers.get("Content-Length")) || 0;
  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    if (total) onProgress(loaded / total);
  }
  return new Blob(chunks).arrayBuffer();
}

// Makes `entry` (of MODELS) the model that runModel uses; onStatus(text) reports the progress.
// Resolves with the backend ({ name: "webgpu" or "wasm", reason }) and the model's class names
// (empty for a model without a classifier).
async function loadModel(entry, onStatus = () => {}) {
  onStatus(`Loading ${entry.label}…`);
  const classNames = entry.classes ? fetch(entry.classes).then((r) => r.json()).then(Object.values) : [];
  const bytes = await download(entry.url, (p) => onStatus(`Downloading ${entry.label} (${Math.round(p * 100)}%)`));
  onStatus(`Initializing ${entry.label}…`);
  const { backend } = await callWorker({ type: "load", model: { id: entry.id, size: entry.size }, bytes }, [bytes]);
  return { backend, classNames: await classNames };
}

// Runs the loaded model on an ImageBitmap of its input size (the bitmap is transferred to the worker).
// features: channels x cells, cellLogits: classes x cells (null without a classifier), cells in
// row-major order of height x width.
const runModel = (bitmap) => callWorker({ type: "run", bitmap }, [bitmap]);

function fillModelMenu(select) {
  select.replaceChildren(...Object.entries(MODEL_GROUPS).map(([group, label]) => {
    const optgroup = Object.assign(document.createElement("optgroup"), { label });
    optgroup.append(...MODELS.filter((m) => m.group === group).map((m) => new Option(`${m.label} (${m.mb} MB)`, m.id)));
    return optgroup;
  }));
}

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("/examples/service-worker.js").catch((err) => console.warn("Service worker:", err));
}
