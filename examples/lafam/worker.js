// imort onnxruntime-web by url into worker
importScripts(
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/ort.min.js"
);
ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/";

const INPUT_WIDTH = 224;
const INPUT_HEIGHT = 224;

let layer4;
let fc;
let fc_wo_pooling; // fc without average-pooling
let results;
let activations;
let output_weights;
let squareClassModel;


onmessage = async (e) => {
  const { data } = e;
  console.log('status', data.status);

  if (data.status === "predict") {
    const imgDataTensor = new ort.Tensor("float32", data.tensor, [
      1,
      3,
      INPUT_HEIGHT,
      INPUT_WIDTH,
    ]);

    activations = await layer4.run({ l_x_: imgDataTensor });
    activations = activations.resnet_layer4_1.cpuData;
    const activationsTensor = new ort.Tensor(
      "float32",
      activations,
      [1, 2048, 7, 7]
    );

    results = await fc.run({ l_activations_: activationsTensor });
    results = results.fc_1.cpuData;
    const predictions = Array.from(softmax(results));
    const heatmap = averageHeatmap(activations, [2048, 7, 7]);

    postMessage({
      status: "results",
      logits: results,
      predictions: predictions,
      heatmap: heatmap,
    });
  }

  if (data.status === "groupmap-bbs") {
    const squaresData = await predict_per_square(data.tensor);
    postMessage({
      status: "square_results_for_groupmap",
      data: squaresData,
    });
  }

  if (data.status === "imagenet-classes") {
    const squaresData = await predict_per_square(data.tensor);
    console.log('classmap done');
    postMessage({
      status: "square_results_for_classmap",
      data: squaresData,
    });
  }

  if (data.status === "heatmap_by_class") {
    let output = new Float32Array(1000).fill(0);
    for (const idx of data.classIdxs) {
      output[idx] = results[idx];
    }
    //w = torch.mm(output, resnet_output_weights)
    const w = matrixMultiply(output, output_weights);
    const weighted_heatmap = averageHeatmap(activations, [2048, 7, 7], w);

    postMessage({
      status: "weighted_heatmap",
      heatmap: weighted_heatmap,
    });
  }

  if (data.status === "class_by_heatmap") {
    let a = new Float32Array(2048 * 7 * 7).fill(0);
    for (let i = 0; i < 2048; i++) {
      for (const idx of data.classIdxs) {
        a[i * 7 * 7 + idx] = activations[i * 7 * 7 + idx];
      }
    }

    let logits = await fc.run({
      l_activations_: new ort.Tensor("float32", a, [1, 2048, 7, 7]),
    });

    logits = logits.fc_1.cpuData;

    const predictions = Array.from(softmax(logits));
    postMessage({
      status: "class_by_heatmap",
      predictions: predictions,
      logits: logits,
    });
  }
};

(async () => {
  layer4 = await ort.InferenceSession.create("resnet50_imagenet_layer4.onnx", {
    executionProviders: ["wasm"],
  });

  fc = await ort.InferenceSession.create("resnet50_imagenet_fc.onnx", {
    executionProviders: ["wasm"],
  });

  fc_wo_pooling = await ort.InferenceSession.create("resnet50_imagenet_fc_wo_averagepooling.onnx", {
    executionProviders: ["wasm"],
  });

  squareClassModel = await ort.InferenceSession.create("resnet_masking_1x1_3x3.onnx", {
    executionProviders: ["wasm"],
  });

  output_weights = await fetch("resnet_output_weights.bin").then((r) =>
    r.arrayBuffer()
  );
  output_weights = new Float32Array(output_weights);

  postMessage({ status: "ready" });
})();

async function predict_per_square(tensor = null) {
  const imgDataTensor = new ort.Tensor("float32", tensor, [1, 3, INPUT_HEIGHT, INPUT_WIDTH,]);
  const results = await squareClassModel.run({ l_x_: imgDataTensor });
  return { logits: results.max_1.cpuData, classIds: results.max_1_1.cpuData };
}

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

function averageHeatmap(arr, shape, weights = null, normalize = true) {
  let heatmap = new Float32Array(shape[1] * shape[2]);
  for (let i = 0; i < shape[1]; i++) {
    for (let j = 0; j < shape[2]; j++) {
      let sum = 0;

      for (let k = 0; k < shape[0]; k++) {
        let w = weights ? weights[k] : 1;
        w = Math.max(w, 0);
        sum += w * arr[k * shape[1] * shape[2] + i * shape[2] + j];
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

function matrixMultiply(r_out, fc_weightsFlat) {
  const n_activations = 2048; // Known/implicit number of columns in fc_weights
  let result = new Float32Array(n_activations).fill(0);
  const n_classes = r_out.length;

  for (let col = 0; col < n_activations; col++) {
    let sum = 0;
    for (let row = 0; row < n_classes; row++) {
      sum += r_out[row] * fc_weightsFlat[row * n_activations + col];
    }
    result[col] = sum;
  }
  return result;
}

function weightHeatmap(w, heatmap) {
  let weighted_heatmap = new Float32Array(49).fill(0);
  for (let i = 0; i < 49; i++) {
    weighted_heatmap[i] = heatmap[i] * w;
  }
  return weighted_heatmap;
}
