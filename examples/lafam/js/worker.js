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

  async calculateEmbedding(data) {
      const { cells } = data;
      const dims = 2048;
      const refEmb = new Float32Array(dims).fill(0);
      
      if (cells.length === 0) return null; // Should not happen
      
      for (let k = 0; k < dims; k++) {
          let sum = 0;
          for (const cellIdx of cells) {
              sum += this.activations[k * 49 + cellIdx];
          }
          refEmb[k] = sum / cells.length;
      }
      
      let norm = 0;
      for(let k=0; k<dims; k++) norm += refEmb[k] * refEmb[k];
      norm = Math.sqrt(norm);
      if(norm > 1e-6) {
          for(let k=0; k<dims; k++) refEmb[k] /= norm;
      }

      // Get prediction label for this embedding (run FC)
      // We construct a 7x7 map where ALL cells have this embedding, just to run through FC easily?
      // Or just reuse evaluate_fc logic. 
      // Actually FC expects [1, 2048, 7, 7]. 
      // We can create a dummy 7x7 filled with this embedding.
      this.activationsBuffer.fill(0);
      for(let k=0; k<dims; k++) {
         const val = refEmb[k];
         // Fill center pixel? Or all? 
         // Global average pooling in ResNet means position doesn't matter for classification if we fill all?
         // ResNet50 usually does GAP before FC. 
         // Looking at architecture: output of layer4 is 7x7. Then GAP -> 2048. Then FC.
         // WAIT. If I have `resnet50_imagenet_fc.onnx`, it probably takes [1, 2048, 7, 7].
         // Does it do GAP inside? 
         // If `predictPerSquare` runs FC on 7x7 spatial, it implies FC is fully convolutional or applied per spatial location?
         // `predictPerSquare` loops 7x7 and constructs a buffer with ONE spatial location filled?
         // See `predictPerSquare`:
         // `this.activationsBuffer[idx] = actsData[idx]` (copies one column)
         // So it creates a volume where mostly zeros, but ONE column is active.
         // And runs FC.
         // So if we want "classification of the object", we should probably do the same or average.
         // Let's just do what `calculateClassByHeatmap` does: fill buffer with masked activations.
         // `calculateClassByHeatmap` fills buffer with `acts` at `classIdxs`.
         // This is effectively "masking" the object. 
         // `logits` returned by that function is the classification of the masked region.
         // So we can just call `calculateClassByHeatmap` logic here or reuse it.
      }
      
      // Actually, let's just get the top prediction from `calculateClassByHeatmap` since `app.js` calls it anyway!
      // `app.js` calls `class_by_heatmap` right after requesting embedding.
      // So `embedding_result` doesn't strictly need the name.
      // BUT `app.js` uses `data.predictionLabel` from `embedding_result` to name the object.
      // So we DO need it.
      
      // Let's implement a quick prediction lookup using the refEmb.
      // We can use the weights `this.output_weights`? No that's for heatmap.
      // We need to run FC.
      
      // Construct input for FC: spread refEmb across 7x7? Or just one pixel?
      // If we fill ONE pixel (e.g. center), and the model expects 7x7, 
      // the GAP will divide by 49?
      // Let's assume standard ResNet GAP = sum / 49.
      // If we have 1 pixel active with value V: GAP = V / 49.
      // If we want the vector to be V, we should probably fill ALL pixels with V.
      // Then GAP = sum(V)/49 = 49*V/49 = V.
      // So let's fill the whole buffer with refEmb.
      
      for(let k=0; k<dims; k++) {
          const val = refEmb[k];
          for(let s=0; s<49; s++) {
              this.activationsBuffer[k*49 + s] = val;
          }
      }
      
      // Run FC
      let res = await this.fc.run({
        l_activations_: new ort.Tensor("float32", this.activationsBuffer, [1, 2048, 7, 7]),
      });
      const logits = res.fc_1.cpuData;
      const maxIdx = logits.indexOf(Math.max(...logits));
      
      // We need class names. `app.js` has them. Worker doesn't have names loaded?
      // `app.js` loads `imagenet_class_index.json`. 
      // Worker just returns label INDEX or Name?
      // Worker output `predictionLabel`... 
      // Worker doesn't have class list.
      // `app.js` has `this.imagenet_classes`.
      // So Worker should return `predictionIdx`. App converts to name.
      
      return {
          status: "embedding_result",
          id: data.id,
          refEmb: refEmb,
          predictionIdx: maxIdx
      };
  }

  async trackObjects(data, tensor) {
      // tensor is already preprocessed and input to layer4
      // Run Inference
      // Note: We might be running this on every frame.
      
      if (!this.layer4) return;
      
      const imgDataTensor = new ort.Tensor("float32", tensor, [1, 3, INPUT_HEIGHT, INPUT_WIDTH]);
      let acts = await this.layer4.run({ l_x_: imgDataTensor });
      const actData = acts.resnet_layer4_1.cpuData; // (1, 2048, 7, 7)
      
      const { objects, settings } = data; // objects: [{id, refEmb, prevX, prevY}], settings: {threshold, ema, resolution}
      const results = [];
      const dims = 2048;
      
      const targetH = settings.resolution;
      const targetW = settings.resolution;
      
      // We need to upscale 7x7 similarity map to target resolution
      // Precompute 224-grid coordinates if not exists?
      // Actually we just compute weighted COM on the upscaled map.
      
      // For each object
      for (const obj of objects) {
          // 1. Compute Similarity (7x7)
          // Dot product: actData(2048, 49) . refEmb(2048) -> sim(49)
          // actData is NC HW: [2048, 49] linearized.
          
          if (!obj.refEmb) continue;
          
          const simMap = new Float32Array(49);
          
          for (let s = 0; s < 49; s++) {
              let dot = 0;
              for (let k = 0; k < dims; k++) {
                  dot += actData[k * 49 + s] * obj.refEmb[k];
              }
              simMap[s] = dot;
          }
          
          // 2. Upsample simMap (7x7) to (Target x Target)
          // We can use a simple bilinear upsampler
          const upscaled = this.upsampleBilinear(simMap, 7, 7, targetW, targetH);
          
          // 3. Threshold & Weighted COM
          let wSum = 0;
          let xSum = 0;
          let ySum = 0;
          const thresh = settings.threshold;
          
          for (let y = 0; y < targetH; y++) {
              for (let x = 0; x < targetW; x++) {
                  let val = upscaled[y * targetW + x];
                  if (val >= thresh) {
                      // val = val; 
                  } else {
                      val = 0; // Hard threshold
                  }
                  
                  if (val > 0) {
                      wSum += val;
                      xSum += val * x;
                      ySum += val * y;
                  }
              }
          }
          
          let resultX = -1;
          let resultY = -1;
          
          if (wSum > 0) {
              // Normalized coords in target resolution
              let cx = xSum / wSum;
              let cy = ySum / wSum;
              
              // Map to input resolution (224x224)
              // If target is 224, it's 1:1.
              // If target is 112, mult by 2.
              const scale = 224 / targetW;
              cx *= scale;
              cy *= scale;
              
              // 4. Smoothing
              if (obj.prevX >= 0 && obj.prevY >= 0 && settings.ema > 0) {
                  const alpha = settings.ema;
                  cx = alpha * obj.prevX + (1 - alpha) * cx;
                  cy = alpha * obj.prevY + (1 - alpha) * cy;
              }
              
              resultX = cx;
              resultY = cy;
          }
          
          results.push({ id: obj.id, x: resultX, y: resultY });
      }
      
      return {
          status: "tracking_results",
          objects: results
      };
  }

  upsampleBilinear(src, srcW, srcH, dstW, dstH) {
      const dst = new Float32Array(dstW * dstH);
      for (let y = 0; y < dstH; y++) {
          for (let x = 0; x < dstW; x++) {
               // Map dst(x,y) to src space
               // Align centers? 
               // Standard: srcX = x * (srcW / dstW)
               const gx = (x + 0.5) * (srcW / dstW) - 0.5;
               const gy = (y + 0.5) * (srcH / dstH) - 0.5;
               
               const gxi = Math.floor(gx);
               const gyi = Math.floor(gy);
               
               // Clamping
               const x0 = Math.max(0, Math.min(srcW - 1, gxi));
               const y0 = Math.max(0, Math.min(srcH - 1, gyi));
               const x1 = Math.max(0, Math.min(srcW - 1, x0 + 1));
               const y1 = Math.max(0, Math.min(srcH - 1, y0 + 1));
               
               const wx = gx - x0;
               const wy = gy - y0;
               
               // Gather 4 samples
               const v00 = src[y0 * srcW + x0];
               const v10 = src[y0 * srcW + x1];
               const v01 = src[y1 * srcW + x0];
               const v11 = src[y1 * srcW + x1];
               
               // Interpolate
               const val = (1 - wy) * ((1 - wx) * v00 + wx * v10) +
                           (wy) * ((1 - wx) * v01 + wx * v11);
                           
               dst[y * dstW + x] = val;
          }
      }
      return dst;
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
const handlers = {
  predict: async (data, tensor) => {
    const res = await model.runPredict(tensor);
    postMessage(res);
  },
  groupmap_bbs: async (data, tensor) => {
    const squaresData = await model.predictPerSquare(tensor);
    postMessage({
      status: "square_results_for_groupmap",
      data: squaresData,
    });
  },
  imagenet_classes: async (data, tensor) => {
    const squaresData = await model.predictPerSquare(tensor);
    postMessage({
      status: "square_results_for_classmap",
      data: squaresData,
    });
  },
  heatmap_by_class: async (data) => {
    const res = model.calculateHeatmapByClass(data.classIdxs);
    postMessage(res);
  },
  class_by_heatmap: async (data) => {
    const res = await model.calculateClassByHeatmap(data.classIdxs);
    postMessage(res);
  },
  calc_embedding: async (data) => {
      const res = await model.calculateEmbedding(data);
      postMessage(res);
  },
  object_tracking: async (data, tensor) => {
    if(!data.objects || data.objects.length === 0) {
      postMessage(await model.runPredict(tensor));
    } else {
      postMessage(await model.trackObjects(data, tensor));
    }
  },
};

onmessage = async (e) => {
  const { data } = e;

  try {
    const handler = handlers[data.status];
    if (handler) {
      let tensor = data.tensor;
      if (data.imageBitmap) {
        tensor = model.preprocessImage(data.imageBitmap);
      }
      await handler(data, tensor);
    }
  } catch (err) {
    console.error("Worker error handling message:", err);
    postMessage({ status: "error", message: err.toString() });
  }
};
