window.addEventListener("error", (event) => {
  debug_log.textContent += event.message + "\n";
  debug_log.scrollTop = debug_log.scrollHeight;
});
window.addEventListener("unhandledrejection", (event) => {
  debug_log.textContent += event.reason + "\n";
  debug_log.scrollTop = debug_log.scrollHeight;
});

const INPUT_WIDTH = 224;
const INPUT_HEIGHT = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const TOP_N = 14;

let video = document.createElement("video");
let predictionList = document.getElementById("prediction-list");

let hidden_canvas = document.createElement("canvas");
let ctx_hidden = hidden_canvas.getContext("2d");

let img_canvas = document.getElementById("img-canvas");
let ctx_img = img_canvas.getContext("2d");

let heatmap_canvas = document.getElementById("heatmap-canvas");
let ctx_heatmap = heatmap_canvas.getContext("2d");

let startButton = document.getElementById("start-button");
let switchCameraButton = document.getElementById("switch-camera-button");

let heatmapOpacity = document.getElementById("heatmap-opacity");
heatmapOpacity.oninput = function () {
  heatmap_canvas.style.opacity = this.value;
};

let debug_devices = document.getElementById("devices");
let debug_log = document.getElementById("log");

async function initCamera() {
  const imagenet_classes = await fetch("./imagenet_class_index.json").then(
    (response) => response.json()
  );

  let cameras = await navigator.mediaDevices.enumerateDevices();
  debug_devices.textContent = JSON.stringify(cameras, null, 2);
  cameras = cameras.filter((device) => device.kind === "videoinput");

  if (cameras.length === 0) {
    startButton.textContent = "No camera";
    return;
  }

  let currentCameraId = cameras[cameras.length - 1].deviceId;
  let localMediaStream = await navigator.mediaDevices.getUserMedia({
    video: {
      deviceId: currentCameraId,
      height: { ideal: 1024 },
    },
  });

  video.srcObject = localMediaStream;

  // video.onloadedmetadata = function (e) {
  //     video.play();
  //     video.pause();
  // };

  // get frame

  // hidden_canvas.width = width;
  // hidden_canvas.height = height;

  // img_canvas.width = min_side;
  // img_canvas.height = min_side;
  // console.log("min_side", min_side);

  const session = await ort.InferenceSession.create(
    "./resnet50_imagenet_modified.onnx",
    { executionProviders: ["wasm"] }
  );

  // let debug_canvas = document.createElement("canvas");
  // debug_canvas.width = min_side;
  // debug_canvas.height = min_side;
  // let ctx_debug = debug_canvas.getContext("2d");
  // document.body.appendChild(debug_canvas);

  if (cameras.length > 1) {
    switchCameraButton.style.display = "block";
    switchCameraButton.onclick = async () => {
      video.pause();
      video.removeEventListener("play", onPlay);
      localMediaStream.getTracks().forEach((track) => track.stop());

      currentCameraId = cameras.find(
        (camera) => camera.deviceId !== currentCameraId
      ).deviceId;

      localMediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: currentCameraId,
          height: { ideal: 1024 },
        },
      });

      video.srcObject = localMediaStream;
      video.addEventListener("play", onPlay, 0);
      video.play();
      startButton.classList.remove("paused");
    };
  }

  let fps = 0;

  video.addEventListener("play", onPlay, 0);

  startButton.disabled = false;

  function onPlay() {
    let $this = this; //cache

    const width = localMediaStream.getVideoTracks()[0].getSettings().width;
    const height = localMediaStream.getVideoTracks()[0].getSettings().height;
    const min_side = Math.min(width, height);
    console.log("min_side", min_side);

    hidden_canvas.width = width;
    hidden_canvas.height = height;
  
    img_canvas.width = min_side;
    img_canvas.height = min_side;
  
    heatmap_canvas.width = min_side;
    heatmap_canvas.height = min_side;

    (async function loop() {
      const start = performance.now();
      if (!$this.paused && !$this.ended) {
        ctx_hidden.drawImage($this, 0, 0);
        let imgData = ctx_hidden.getImageData(0, 0, width, height);

        // let imgDataArray = new Float32Array(3 * height * width);
        // for (i = 0; i < imgData.data.length; i = i + 4) {
        //     imgDataArray[i / 4] = imgData.data[i]; // R
        //     imgDataArray[i / 4 + 1] = imgData.data[i + 1]; // G
        //     imgDataArray[i / 4 + 2] = imgData.data[i + 2]; // B
        // }

        const croppedFrame = ImageProcessor.fromImageData(imgData).squareCrop();

        // ctx_img.putImageData(new ImageData(ImageProcessor.toImageData(transformd_img),
        //     transformd_img.width, transformd_img.height), 0, 0);

        const transformed_img = croppedFrame
          .resize(INPUT_WIDTH, INPUT_HEIGHT, "bilinear")
          .normalize(MEAN, STD);

        // let d_img = ImageProcessor.toImageData(transformd_img.demoralize(MEAN, STD));
        // ctx_debug.putImageData(new ImageData(d_img, transformd_img.width, transformd_img.height), 0, 0);

        let imgDataTensor = new ort.Tensor(
          "float32",
          ImageProcessor.toTensor(transformed_img),
          [1, 3, INPUT_HEIGHT, INPUT_WIDTH]
        );

        let feeds = { l_x_: imgDataTensor };
        window.results = await session.run(feeds);
        let output = Array.from(softmax(results.fc_1.cpuData));
        const top_n_idx = argmax_top_n(output, TOP_N);
        // console.log('top_n_idx', top_n_idx);

        // console.log('classes', top_n_idx.map(idx => imagenet_classes[idx]));
        // console.log('probabilities', top_n_idx.map(idx => output[idx]));

        updatePredictionList(
          top_n_idx.map((idx) => imagenet_classes[idx]),
          top_n_idx.map((idx) => output[idx])
        );

        // console.log(results.layer4_1.cpuData);
        // console.log('min', Math.min(...results.layer4_1.cpuData))
        // console.log('max', Math.max(...results.layer4_1.cpuData))
        // sum
        // console.log('sum', results.layer4_1.cpuData.reduce((a, b) => a + b, 0));

        // let heatmap = new ImageProcessor([results.layer4_1.cpuData], 7, 7)
        //     .resize(min_side, min_side, 'nearest')
        // .demoralize([0, 0, 0], [1, 1, 1]);
        let heatmap = averageHeatmap(results.layer4_1.cpuData, [2048, 7, 7]);
        heatmap = mapToPallete(heatmap, viridis);
        // console.log('heatmap', heatmap);
        heatmap = new ImageProcessor(heatmap, 7, 7).resize(
          min_side,
          min_side,
          "nearest"
        );
        // console.log('heatmap', heatmap);

        // const blended = blendHeatmap(croppedFrame, heatmap, 0.7);

        // ctx_debug.putImageData(new ImageData(ImageProcessor.toImageData(heatmap), min_side, min_side), 0, 0);
        ctx_img.putImageData(
          new ImageData(
            ImageProcessor.toImageData(croppedFrame),
            min_side,
            min_side
          ),
          0,
          0
        );

        ctx_heatmap.putImageData(
          new ImageData(
            ImageProcessor.toImageData(heatmap),
            min_side,
            min_side
          ),
          0,
          0
        );


        // console.log(1000 / (performance.now() - start));

        setTimeout(loop, 0);
      }
    })();
  }
}

// function blendHeatmap(img, heatmap, opacity) {
//   const length = img.width * img.height;
//   let blended = new ImageProcessor(
//     [
//       new Float32Array(length),
//       new Float32Array(length),
//       new Float32Array(length),
//     ],
//     img.width,
//     img.height
//   );
//   for (let i = 0; i < length; i++) {
//     for (let c = 0; c < img.channels.length; c++) {
//       blended.channels[c][i] =
//         img.channels[c][i] * (1 - opacity) + heatmap.channels[c][i] * opacity;
//     }
//   }
//   return blended;
// }

function mapToPallete(x, pallete) {
  let heatmap = [
    new Float32Array(x.length),
    new Float32Array(x.length),
    new Float32Array(x.length),
  ];
  for (let i = 0; i < x.length; i++) {
    const color = pallete[Math.round(x[i] * (pallete.length - 1))];
    heatmap[0][i] = color[0];
    heatmap[1][i] = color[1];
    heatmap[2][i] = color[2];
  }
  return heatmap;
}

const viridis = [
  [68, 1, 84],
  [72, 35, 116],
  [64, 67, 135],
  [52, 94, 141],
  [41, 120, 142],
  [32, 144, 140],
  [34, 167, 132],
  [68, 190, 112],
  [121, 209, 81],
  [189, 222, 38],
  [253, 231, 36],
];
const inferno = [
  [0, 0, 4],
  [31, 12, 72],
  [85, 15, 109],
  [136, 34, 106],
  [186, 54, 85],
  [227, 89, 51],
  [249, 140, 10],
  [249, 201, 50],
  [248, 252, 117],
];

class ImageProcessor {
  constructor(channels, width, height) {
    this.channels = channels;
    this.width = width;
    this.height = height;
  }

  static fromImageData(imageData) {
    let img = new ImageProcessor();
    img.width = imageData.width;
    img.height = imageData.height;
    img.channels = ImageProcessor.toNdarray(imageData, 3);

    return img;
  }

  static toTensor(img) {
    let tensor = new Float32Array(img.channels.length * img.width * img.height);
    for (let i = 0; i < img.channels.length; i++) {
      tensor.set(img.channels[i], i * img.width * img.height);
    }
    return tensor;
  }

  squareCrop() {
    const size = Math.min(this.width, this.height);
    const startX = Math.round((this.width - size) / 2);
    const startY = Math.round((this.height - size) / 2);
    const endY = startY + size;

    let channels = [];
    for (let i = 0; i < this.channels.length; i++) {
      channels.push(new Float32Array(size * size));
    }

    let croppedIndex = 0;

    for (let y = startY; y < endY; y++) {
      const rowStart = y * this.width + startX;
      const rowEnd = rowStart + size;

      for (let channel = 0; channel < this.channels.length; channel++) {
        channels[channel].set(
          this.channels[channel].slice(rowStart, rowEnd),
          croppedIndex
        );
      }

      croppedIndex += size;
    }

    return new ImageProcessor(channels, size, size);
  }

  resize(newWidth, newHeight, interpolation = "bilinear") {
    const resized = [];
    for (let i = 0; i < this.channels.length; i++) {
      resized.push(new Float32Array(newWidth * newHeight));
    }

    const xRatio = this.width / newWidth;
    const yRatio = this.height / newHeight;

    for (let y = 0; y < newHeight; y++) {
      for (let x = 0; x < newWidth; x++) {
        if (interpolation === "nearest") {
          this._nearestNeighbor(x, y, xRatio, yRatio, resized, newWidth);
        } else if (interpolation === "bilinear") {
          this._bilinearInterpolation(x, y, xRatio, yRatio, resized, newWidth);
        } else {
          throw new Error(
            `interpolation method ${interpolation} is not supported.`
          );
        }
      }
    }

    return new ImageProcessor(resized, newWidth, newHeight);
  }

  normalize(mean, std) {
    // Validate mean and std have 3 elements each
    if (mean.length !== 3 || std.length !== 3) {
      throw new Error("Mean and standard deviation must each have 3 elements.");
    }

    const length = this.width * this.height;
    const n_channels = this.channels.length;
    let channels = [
      new Float32Array(length),
      new Float32Array(length),
      new Float32Array(length),
    ];

    for (let i = 0; i < length; i++) {
      // Normalize each channel
      for (let channel = 0; channel < n_channels; channel++) {
        channels[channel][i] =
          (this.channels[channel][i] / 255 - mean[channel]) / std[channel];
      }
    }

    return new ImageProcessor(channels, this.width, this.height);
  }

  demoralize(mean, std) {
    // Validate mean and std have 3 elements each
    if (mean.length !== 3 || std.length !== 3) {
      throw new Error("Mean and standard deviation must each have 3 elements.");
    }

    const length = this.width * this.height;
    let channels = [
      new Float32Array(length),
      new Float32Array(length),
      new Float32Array(length),
    ];

    for (let i = 0; i < length; i++) {
      // Denormalize each channel
      for (let channel = 0; channel < 3; channel++) {
        channels[channel][i] =
          (this.channels[channel][i] * std[channel] + mean[channel]) * 255;
      }
    }

    return new ImageProcessor(channels, this.width, this.height);
  }

  static toImageData(img) {
    const length = img.width * img.height;
    let rgba = new Uint8ClampedArray(length * 4);
    for (let i = 0; i < length * 4; i += 4) {
      for (let channel = 0; channel < img.channels.length; channel++) {
        rgba[i + channel] = img.channels[channel][i / 4];
      }
      rgba[i + 3] = 255;
    }

    return rgba;
  }

  static toNdarray(imageData, n_channels) {
    const length = imageData.width * imageData.height;
    let channels = [];
    for (let i = 0; i < n_channels; i++) {
      channels.push(new Float32Array(length));
    }

    for (let i = 0; i < imageData.data.length; i += 4) {
      for (let channel = 0; channel < n_channels; channel++) {
        channels[channel][i / 4] = imageData.data[i + channel];
      }
      // ignore [i + 4] as it is alpha channel
    }
    // 3. Concatenate RGB to transpose [224, 224, 3] -> [3, 224, 224] to a number array
    // const transposedData = redArray.concat(greenArray).concat(blueArray);

    return channels;
  }

  _nearestNeighbor(x, y, xRatio, yRatio, output, newWidth) {
    const nearestX = Math.floor((x + 0.5) * xRatio);
    const nearestY = Math.floor((y + 0.5) * yRatio);
    const idxSrc = nearestY * this.width + nearestX;
    const idxDest = y * newWidth + x;

    for (let channel = 0; channel < this.channels.length; channel++) {
      output[channel][idxDest] = this.channels[channel][idxSrc];
    }
  }

  _bilinearInterpolation(x, y, xRatio, yRatio, output, newWidth) {
    const xL = Math.floor(x * xRatio);
    const yL = Math.floor(y * yRatio);
    const xH = Math.ceil(x * xRatio);
    const yH = Math.ceil(y * yRatio);
    const xWeight = x * xRatio - xL;
    const yWeight = y * yRatio - yL;
    const idxDest = y * newWidth + x;

    for (let channel = 0; channel < 3; channel++) {
      const valTL = this.channels[channel][yL * this.width + xL];
      const valTR = this.channels[channel][yL * this.width + xH];
      const valBL = this.channels[channel][yH * this.width + xL];
      const valBR = this.channels[channel][yH * this.width + xH];

      const top = valTL + (valTR - valTL) * xWeight;
      const bottom = valBL + (valBR - valBL) * xWeight;
      output[channel][idxDest] = top + (bottom - top) * yWeight;
    }
  }
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

function argmax_top_n(arr, n) {
  let indices = arr.map((e, i) => i);
  indices.sort((a, b) => arr[b] - arr[a]);
  return indices.slice(0, n);
}

function toggleVideo(el) {
  if (video.paused) {
    video.play();
    el.classList.remove("paused");
  } else {
    video.pause();
    el.classList.add("paused");
  }
}

function updatePredictionList(classes, probabilities) {
  // create list with progress bars
  predictionList.innerHTML = "";
  for (let i = 0; i < classes.length; i++) {
    let label = document.createElement("label");
    label.textContent = `${classes[i]}: ${Math.round(probabilities[i] * 100)}%`;
    predictionList.appendChild(label);

    let progress = document.createElement("progress");
    progress.value = probabilities[i];
    progress.max = 1;
    predictionList.appendChild(progress);
  }
}

// C*H*W to H*W array
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

initCamera();
