importScripts(
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/ort.min.js"
);
ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/";

const INPUT_WIDTH = 224;
const INPUT_HEIGHT = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

class LaFAMModel {
  constructor() {
    this.layer4 = null;
    this.fc = null;
    this.results = null;
    this.activations = null;
    this.output_weights = null;

    this.offscreenCanvas = new OffscreenCanvas(INPUT_WIDTH, INPUT_HEIGHT);
    this.ctx = this.offscreenCanvas.getContext("2d", { willReadFrequently: true });
    
    this.imageBufferLength = INPUT_WIDTH * INPUT_HEIGHT;
    this.preprocessedData = new Float32Array(3 * this.imageBufferLength);
    
    this.activationsBuffer = new Float32Array(2048 * 7 * 7);
  }

  async init() {
    try {
      this.layer4 = await ort.InferenceSession.create(
        "../assets/models/resnet50_imagenet_layer4.onnx",
        { executionProviders: ["wasm"] }
      );

      this.fc = await ort.InferenceSession.create(
        "../assets/models/resnet50_imagenet_fc.onnx",
        { executionProviders: ["wasm"] }
      );

      const response = await fetch("../assets/models/resnet_output_weights.bin");
      const buffer = await response.arrayBuffer();
      this.output_weights = new Float32Array(buffer);

      return { status: "ready" };
    } catch (error) {
      console.error("Model initialization error:", error);
      return { 
        status: "error", 
        message: "Failed to load ML models: " + error.message 
      };
    }
  }

  preprocessImage(imageBitmap) {
    const w = INPUT_WIDTH;
    const h = INPUT_HEIGHT;

    // Resize using OffscreenCanvas
    this.ctx.drawImage(imageBitmap, 0, 0, w, h);
    const imageData = this.ctx.getImageData(0, 0, w, h);
    
    // Normalize and convert to NCHW tensor
    const length = this.imageBufferLength;
    const data = imageData.data;
    const float32Data = this.preprocessedData; // Use pre-allocated buffer
    
    for (let i = 0; i < length; i++) {
        // Red
        float32Data[i] = (data[i * 4] / 255.0 - MEAN[0]) / STD[0];
        // Green
        float32Data[i + length] = (data[i * 4 + 1] / 255.0 - MEAN[1]) / STD[1];
        // Blue
        float32Data[i + 2 * length] = (data[i * 4 + 2] / 255.0 - MEAN[2]) / STD[2];
    }
    
    imageBitmap.close(); 
    return float32Data;
  }

  async runPredict(tensor) {
    const imgDataTensor = new ort.Tensor("float32", tensor, [1, 3, INPUT_HEIGHT, INPUT_WIDTH]);

    let activationsResult = await this.layer4.run({ l_x_: imgDataTensor });
    this.activations = activationsResult.resnet_layer4_1.cpuData;
    
    const activationsTensor = new ort.Tensor("float32", this.activations, [1, 2048, 7, 7]);

    let resultsResult = await this.fc.run({ l_activations_: activationsTensor });
    this.results = resultsResult.fc_1.cpuData;
    
    const predictions = Array.from(this.softmax(this.results));
    const heatmap = this.averageHeatmap(this.activations, [2048, 7, 7]);

    return {
      status: "results",
      logits: this.results,
      predictions: predictions,
      heatmap: heatmap,
    };
  }

  async predictPerSquare(tensor) {
    const imgDataTensor = new ort.Tensor("float32", tensor, [1, 3, INPUT_HEIGHT, INPUT_WIDTH]);
    
    let acts = await this.layer4.run({ l_x_: imgDataTensor });
    // Acts is actually a TypedArray from cpuData, likely Float32Array
    const actsData = acts.resnet_layer4_1.cpuData;

    let logits = new Float32Array(7 * 7).fill(0);
    let classIds = new Uint16Array(7 * 7).fill(0);
        
    const PATCH_SIZE = 2048;
    const SPATIAL = 49; // 7*7
        
    for (let i = 0; i < 7; i++) {
        for (let j = 0; j < 7; j++) {
            this.activationsBuffer.fill(0);
            
            const spatialOffset = i * 7 + j;
            for (let k = 0; k < 2048; k++) {
                const idx = k * 49 + spatialOffset;
                this.activationsBuffer[idx] = actsData[idx];
            }

            let r = await this.fc.run({
                l_activations_: new ort.Tensor("float32", this.activationsBuffer, [1, 2048, 7, 7]),
            });

            const logit = r.fc_1.cpuData;
            const maxIdx = logit.indexOf(Math.max(...logit));
            logits[spatialOffset] = logit[maxIdx];
            classIds[spatialOffset] = maxIdx;
        }
    }

    return { logits: logits, classIds: classIds };
  }

  calculateHeatmapByClass(classIdxs) {
    let output = new Float32Array(1000).fill(0);
    for (const idx of classIdxs) {
      output[idx] = this.results[idx];
    }
    
    const w = this.matrixMultiply(output, this.output_weights);
    const weighted_heatmap = this.averageHeatmap(this.activations, [2048, 7, 7], w);

    return {
      status: "weighted_heatmap",
      heatmap: weighted_heatmap,
    };
  }

  async calculateClassByHeatmap(classIdxs) {
    this.activationsBuffer.fill(0);
    const buffer = this.activationsBuffer;
    const acts = this.activations;
    
    for (let i = 0; i < 2048; i++) {
        const start = i * 49;
        for (const idx of classIdxs) {
            buffer[start + idx] = acts[start + idx];
        }
    }

    let logits = await this.fc.run({
      l_activations_: new ort.Tensor("float32", buffer, [1, 2048, 7, 7]),
    });

    logits = logits.fc_1.cpuData;
    const predictions = Array.from(this.softmax(logits));
    
    return {
        status: "class_by_heatmap",
        predictions: predictions,
        logits: logits
    };
  }

  // Utilities
  softmax(arr) {
    const max = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(exp => exp / sum);
  }

  averageHeatmap(arr, shape, weights = null, normalize = true) {
    const C = shape[0];
    const H = shape[1];
    const W = shape[2];
    const spatialSize = H * W; 
    
    let heatmap = new Float32Array(spatialSize);
    
    for (let i = 0; i < H; i++) {
      for (let j = 0; j < W; j++) {
        let sum = 0;
        const spatialIdx = i * W + j;
        
        for (let k = 0; k < C; k++) {
            let w = weights ? weights[k] : 1;
            w = Math.max(w, 0);
            sum += w * arr[k * spatialSize + spatialIdx];
        }
        heatmap[spatialIdx] = sum / C;
      }
    }
    
    if (normalize) {
      let max = -Infinity;
      let min = Infinity;
      for(let i=0; i<heatmap.length; i++) {
          if (heatmap[i] > max) max = heatmap[i];
          if (heatmap[i] < min) min = heatmap[i];
      }
      
      const range = max - min + 1e-6;
      for(let i=0; i<heatmap.length; i++) {
          heatmap[i] = (heatmap[i] - min) / range;
      }
    }
    return heatmap;
  }

  matrixMultiply(r_out, fc_weightsFlat) {
    const n_activations = 2048;
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
}

// Global instance
const model = new LaFAMModel();

// Initialize immediately
model.init().then((msg) => postMessage(msg));

// Message Handler
onmessage = async (e) => {
  const { data } = e;
  
  try {
      if (data.status === "predict" || data.status === "groupmap-bbs" || data.status === "imagenet-classes") {
          let tensor = data.tensor;
          if (data.imageBitmap) {
             tensor = model.preprocessImage(data.imageBitmap);
          }
          
          if (data.status === "predict") {
              const res = await model.runPredict(tensor);
              postMessage(res);
          } else if (data.status === "groupmap-bbs") {
              const squaresData = await model.predictPerSquare(tensor);
              postMessage({
                  status: "square_results_for_groupmap",
                  data: squaresData,
              });
          } else if (data.status === "imagenet-classes") {
              const squaresData = await model.predictPerSquare(tensor);
              postMessage({
                  status: "square_results_for_classmap",
                  data: squaresData,
              });
          }
      } 
      else if (data.status === "heatmap_by_class") {
          const res = model.calculateHeatmapByClass(data.classIdxs);
          postMessage(res);
      } 
      else if (data.status === "class_by_heatmap") {
          const res = await model.calculateClassByHeatmap(data.classIdxs);
          postMessage(res);
      }
  } catch(err) {
      console.error("Worker error handling message:", err);
      postMessage({status: "error", message: err.toString()});
  }
};
