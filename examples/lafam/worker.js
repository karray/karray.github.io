// imort onnxruntime-web by url into worker
importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js");
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

let session;

const INPUT_WIDTH = 224;
const INPUT_HEIGHT = 224;

onmessage = async (e) => {
  const { data } = e;
  const imgDataTensor = new ort.Tensor(
    "float32",
    data,
    [1, 3, INPUT_HEIGHT, INPUT_WIDTH]
  );

  const results = await session.run({ l_x_: imgDataTensor });
  const predictions = Array.from(softmax(results.fc_1.cpuData));
  const heatmap = averageHeatmap(results.layer4_1.cpuData, [2048, 7, 7]);

  postMessage({ status: "results", predictions: predictions, heatmap: heatmap });
};

(async () => {
  session = await ort.InferenceSession.create(
    "resnet50_imagenet_modified.onnx",
    { executionProviders: ["wasm"] }
  );

  postMessage({ status: "ready" });
})();

function softmax(arr) {
  return arr.map(function (value, index) {
    return (
      Math.exp(value) /
      arr
        .map(function (y) {
          return Math.exp(y);
        })
        .reduce(function (a, b) {
          return a + b;
        })
    );
  });
}

function averageHeatmap(arr, shape, normalize = true) {
  let heatmap = new Float32Array(shape[1] * shape[2]);
  for (let i = 0; i < shape[1]; i++) {
    for (let j = 0; j < shape[2]; j++) {
      let sum = 0;
      for (let k = 0; k < shape[0]; k++) {
        sum += arr[k * shape[1] * shape[2] + i * shape[2] + j];
      }
      heatmap[i * shape[2] + j] = sum / shape[0];
    }
  }
  if (normalize) {
    let max = Math.max(...heatmap);
    let min = Math.min(...heatmap);
    heatmap = heatmap.map((x) => (x - min) / (max - min + 1e-6));
  }
  return heatmap;
}
