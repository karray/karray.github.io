import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

let session = null;
let model = null;

const backend = (async () => {
  if (!navigator.gpu) return { name: "wasm", reason: "this browser offers no WebGPU" };
  const adapter = await navigator.gpu.requestAdapter().catch(() => null);
  if (!adapter) return { name: "wasm", reason: "WebGPU found no GPU (on Linux, Chrome needs Vulkan: chrome://flags/#enable-vulkan)" };
  if (adapter.info?.isFallbackAdapter ?? adapter.isFallbackAdapter) {
    return { name: "wasm", reason: `WebGPU only offers a software adapter (${adapter.info?.architecture || "unknown"})` };
  }
  return { name: "webgpu", reason: `${adapter.info?.vendor || ""} ${adapter.info?.architecture || ""}`.trim() };
})();

// ImageBitmap of the model's input size -> NCHW tensor in [0, 1] (each model normalizes it itself)
function preprocess(bitmap) {
  const size = model.size;
  const ctx = new OffscreenCanvas(size, size).getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, size, size);
  bitmap.close();
  const rgba = ctx.getImageData(0, 0, size, size).data;
  const n = size * size;
  const data = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) data[c * n + i] = rgba[4 * i + c] / 255;
  }
  return new ort.Tensor("float32", data, [1, 3, size, size]);
}

const handlers = {
  // { model, bytes }: bytes is the downloaded .onnx file
  async load({ model: entry, bytes }) {
    const ep = await backend;
    await session?.release();
    session = null;
    session = await ort.InferenceSession.create(bytes, { executionProviders: ep.name === "webgpu" ? ["webgpu", "wasm"] : ["wasm"] });
    // the first run compiles the GPU shaders, which can take seconds: do it while the model is loading
    const { size } = entry;
    await session.run({ image: new ort.Tensor("float32", new Float32Array(3 * size * size), [1, 3, size, size]) });
    model = entry;
    return { backend: ep };
  },
  async run({ bitmap }) {
    const out = await session.run({ image: preprocess(bitmap) });
    const [, channels, height, width] = out.features.dims;
    return {
      features: (await out.features.getData()).slice(),       // copies, so that they can be transferred
      cellLogits: out.cell_logits ? (await out.cell_logits.getData()).slice() : null,   // null: no classifier
      logits: out.logits ? (await out.logits.getData()).slice() : null,   // null: the mean of the cells is the prediction
      channels, classes: out.cell_logits?.dims[1] ?? 0, height, width,
    };
  },
};

async function handle(data) {
  try {
    const result = await handlers[data.type](data);
    const transfer = [result.features?.buffer, result.cellLogits?.buffer, result.logits?.buffer].filter(Boolean);
    postMessage({ id: data.id, ...result }, transfer);
  } catch (err) {
    console.error(err);
    postMessage({ id: data.id, error: String(err) });
  }
}

let queue = Promise.resolve();
onmessage = ({ data }) => (queue = queue.then(() => handle(data)));
