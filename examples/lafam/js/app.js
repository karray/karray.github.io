window.addEventListener("error", (event) => {
  addLogMsg(event.message);
});
window.addEventListener("unhandledrejection", (event) => {
  addLogMsg(event.reason);
});

function addLogMsg(msg) {
  debug_log.textContent += msg + "\n";
  debug_log.scrollTop = debug_log.scrollHeight;
}

const debug_devices = document.getElementById("devices");
const debug_log = document.getElementById("log");

const INPUT_WIDTH = 224;
const INPUT_HEIGHT = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const TOP_N = 14;

// Application state management
const AppState = {
  includeGroups: null,
  excludeGroups: null,
  grouper: null,
  selectionEnabled: false,
  currentImage: null,
  squareData: null,
  isWorkerReady: false,
  trackingObjects: [],
  activeTrackingObjectId: null,
  trackingSettings: {
    threshold: 0.5,
    ema: 0.75,
    overlayAlpha: 0.5,
    resolution: 224
  }
};


class ModelWorker {
  constructor(url) {
    let $this = this;
    this.initCamera();

    this.worker = new window.Worker(url);
    this.worker.onmessage = (e) => {
      $this._onmessage(e);
    };

    this.palettes;
    this.currentPalette;



    this.initElements();
    this.initEvents();
    this._clearGroupMapDisplay();

    // Update loading text
    const loadingText = document.getElementById("loading-text");
    
    // Parallelize data fetches with per-file progress tracking
    const dataFiles = [
      { name: "ImageNet classes", url: "assets/data/imagenet_class_index.json", loaded: false },
      { name: "Color palettes", url: "assets/data/palettes.json", loaded: false },
      { name: "Exclude groups", url: "assets/data/exclude_groups.json", loaded: false },
      { name: "Include groups", url: "assets/data/include_groups.json", loaded: false }
    ];

    const updateLoadingDisplay = () => {
      if (loadingText) {
        const status = dataFiles.map(f => 
          `${f.loaded ? '✓' : '○'} ${f.name}`
        ).join('\n');
        loadingText.innerHTML = status.replace(/\n/g, '<br>');
      }
    };

    // Initial display
    updateLoadingDisplay();

    const fetchWithProgress = (file) => {
      return fetch(file.url)
        .then(r => r.json())
        .then(data => {
          file.loaded = true;
          updateLoadingDisplay();
          return data;
        });
    };

    Promise.all(dataFiles.map(fetchWithProgress))
      .then(([imagenetClasses, palettes, excludeGroups, includeGroups]) => {
        this.imagenet_classes = imagenetClasses;
        this.palettes = palettes;
        this.currentPalette = Object.keys(palettes)[0];
        AppState.excludeGroups = excludeGroups;
        AppState.includeGroups = includeGroups;
        
        if (loadingText) {
          loadingText.textContent = "Initializing model...";
        }
        
        // Initialize palette select options
        for (const palette in palettes) {
          const option = document.createElement("option");
          option.value = palette;
          option.textContent = palette;
          this.paletteSelect.appendChild(option);
        }
      }).catch(error => {
        console.error("Failed to load data files:", error);
        addLogMsg("Error loading data files: " + error.message);
        if (loadingText) {
          loadingText.textContent = "Error loading data files!";
        }
      });
  }

  initElements() {
    this.paletteSelect = document.getElementById("palette-select");
    this.modeSelect = document.getElementById("mode-select");
    this.mainSection = document.getElementById("main-section");
    this.video = document.createElement("video");
    this.predictionList = document.getElementById("prediction-list");
    this.hidden_canvas = document.createElement("canvas");
    this.ctx_hidden = this.hidden_canvas.getContext("2d");
    this.img_canvas = document.getElementById("img-canvas");
    this.ctx_img = this.img_canvas.getContext("2d");
    this.heatmap_canvas = document.getElementById("heatmap-canvas");
    this.ctx_heatmap = this.heatmap_canvas.getContext("2d");
    this.bb_canvas = document.getElementById("bounding-boxes-canvas");
    this.startButton = document.getElementById("start-button");
    this.switchCameraButton = document.getElementById("switch-camera-button");
    this.heatmapOpacity = document.getElementById("heatmap-opacity");
    this.predefinedFiles = document.getElementById("predefined-files");
    this.uploadButton = document.getElementById("upload-button");
    this.uploadInput = document.getElementById("upload-input");
    this.heatmapGrid = document.getElementById("grid");
    this.clearSelection = document.getElementById("clear-selection");

    // Object Tracking Elements
    this.trackingControls = document.getElementById("tracking-controls");
    this.addObjectBtn = document.getElementById("add-object-btn");
    this.trackingObjectsList = document.getElementById("tracking-objects-list");
    this.trackingResolution = document.getElementById("tracking-resolution");
    this.trackingThreshold = document.getElementById("tracking-threshold");
    this.trackingThresholdVal = document.getElementById("tracking-threshold-val");
    this.trackingEma = document.getElementById("tracking-ema");
    this.trackingEmaVal = document.getElementById("tracking-ema-val");
    this.trackingOverlayAlpha = document.getElementById("tracking-overlay-alpha");
    this.trackingOverlayAlphaVal = document.getElementById("tracking-overlay-alpha-val");
  }

  initEvents() {
    const $this = this;

    this.video.onloadedmetadata = async () => {
      $this.setSize($this.video.videoWidth, $this.video.videoHeight);
      $this.startButton.disabled = false;

      if ($this.video.srcObject) {
        try {
          await $this.video.play();
          AppState.currentImage = await $this._getFrame();
          if($this.mainSection.classList.contains("paused")){
            $this.video.pause();
          }
        } catch (err) {
          addLogMsg("Failed to get camera preview frame: " + err);
          console.warn("Failed to get camera preview frame: " + err);
        }
      } else {
        AppState.currentImage = await $this._getFrame();
      }
      if(AppState.isWorkerReady){
        $this._postMessage();
      }
    };

    this.predefinedFiles.addEventListener("change", () => {
      $this._clearSelections();
      $this._clearGroupMapDisplay();
      $this.video.pause();
      $this.mainSection.classList.add("paused");
      AppState.squareData = null;
      const status = this.modeSelect.value;
      $this.load_selected_image(status);
    });

    this.uploadButton.addEventListener("click", () => {
      this.uploadInput.click();
    });

    this.uploadInput.addEventListener("change", async (e) => {      
      $this._clearSelections();
      $this._clearGroupMapDisplay();
      $this.video.pause();
      $this.mainSection.classList.add("paused");
      AppState.squareData = null;

      const file = e.target.files && e.target.files[0];
      if (!file) return;

      this._handleUploadedBlob(file, this.modeSelect.value);

      if ($this.uploadInput.files.length > 0) {
        $this.predefinedFiles.value = "";
      }
    });

    this.predictionList.addEventListener("click", (event) => {
      if (!AppState.selectionEnabled) return;
      if (!this.video.paused) return;
      const div = event.target.closest(".prediction");
      if (!div) return; // Clicked outside a prediction div

      let selectedIdxs = [];
      if (!div.classList.contains("selected")) {
        selectedIdxs.push(parseInt(div.getAttribute("data-idx")));
        div.classList.add("selected");
      } else {
        div.classList.remove("selected");
      }

      for (const el of document.querySelectorAll(".prediction.selected")) {
        if (el !== div) {
          selectedIdxs.push(parseInt(el.getAttribute("data-idx")));
        }
      }

      if (selectedIdxs.length > 0) {
        this.clearSelection.disabled = false;
        this.worker.postMessage({
          status: "heatmap_by_class",
          classIdxs: selectedIdxs,
        });
      } else {
        this.clearSelection.disabled = true;
        this.updateResults(
          this.results.heatmap,
          this.results.predictions,
          this.results.logits
        );
      }
    });

    this.clearSelection.addEventListener("click", () => {
      this._clearSelections();
      this.updateResults(
        this.results.heatmap,
        this.results.predictions,
        this.results.logits
      );
    });

    this.heatmapOpacity.oninput = () => {
      $this.heatmap_canvas.style.opacity = $this.heatmapOpacity.value;
    };

    this.paletteSelect.onchange = function () {
      if (!AppState.selectionEnabled) return;
      $this.currentPalette = this.value;
      $this.updateHeatmap($this.currentHeatmap);
    };

    this.startButton.addEventListener("click", async () => {
      AppState.squareData = null;

      const hasStream = !!this.video.srcObject;
      const hasFileSrc = !!this.video.currentSrc; // populated after src is set and load starts

      if (!hasStream && !hasFileSrc) {
        console.warn("Cannot play: video has no srcObject and no src/currentSrc.");
        return;
      }

      if (this.video.paused || this.video.ended) {
        this._clearSelections();
        this.mainSection.classList.remove("paused");

        try {
          await this.video.play();
          this._updateVideo();
        } catch (err) {
          console.warn("video.play() failed:", err);

          if (this.video.srcObject && typeof this.video.srcObject.getVideoTracks === "function") {
            console.log(
              "track states:",
              this.video.srcObject.getVideoTracks().map(t => ({ readyState: t.readyState, enabled: t.enabled, muted: t.muted }))
            );
          }
        }
      } else {
        this.video.pause();
        this.mainSection.classList.add("paused");
      }
    });

    this.modeSelect.addEventListener("change", (e) => {
      this.processModeChange();
    });

    // Tracking UI Events
    this.addObjectBtn.addEventListener("click", () => {
        this.addTrackingObject();
    });

    this.trackingResolution.addEventListener("change", (e) => {
        AppState.trackingSettings.resolution = parseInt(e.target.value);
    });

    this.trackingThreshold.addEventListener("input", (e) => {
        AppState.trackingSettings.threshold = parseFloat(e.target.value);
        this.trackingThresholdVal.innerText = parseFloat(e.target.value).toFixed(2);
    });

    this.trackingEma.addEventListener("input", (e) => {
        AppState.trackingSettings.ema = parseFloat(e.target.value);
        this.trackingEmaVal.innerText = parseFloat(e.target.value).toFixed(2);
    });

    this.trackingOverlayAlpha.addEventListener("input", (e) => {
        AppState.trackingSettings.overlayAlpha = parseFloat(e.target.value);
        this.trackingOverlayAlphaVal.innerText = parseFloat(e.target.value).toFixed(2);
    });

    // root event listener for cells (divs)
    this.heatmapGrid.addEventListener("click", (e) => {
      if (!AppState.selectionEnabled) return;

      const div = e.target.closest("div");
      if (!div) return;

      if (this.modeSelect.value === "object_tracking") {
          // Object Tracking Mode
          if (AppState.activeTrackingObjectId === null) return;
          
          let obj = AppState.trackingObjects.find(o => o.id === AppState.activeTrackingObjectId);
          if (!obj) return;

          const idx = parseInt(div.getAttribute("data-idx"));
          const cellIndex = obj.cells.indexOf(idx);
          
          if (cellIndex === -1) {
              obj.cells.push(idx);
              div.classList.add("selected");
              div.style.borderColor = obj.color;
              div.style.boxShadow = `0 0 20px ${obj.color} inset`;
          } else {
              obj.cells.splice(cellIndex, 1);
              div.classList.remove("selected");
              div.style.borderColor = "";
              div.style.boxShadow = "";
          }

          // Request new embedding if cells exist
          if (obj.cells.length > 0) {
              this.worker.postMessage({
                  status: "calc_embedding",
                  id: obj.id,
                  cells: obj.cells,
                  imageBitmap: AppState.currentImage // Needs current image for ref aggregation
              });
              
              // Also show heatmap for feedback (like classification mode)
              this.worker.postMessage({
                  status: "class_by_heatmap",
                  classIdxs: obj.cells,
              });
          } else {
              // Clear predictions/heatmap if no cells
              this.updateResults(this.results.heatmap, this.results.predictions, this.results.logits);
          }

      } else {
          // Standard Modes
            this._clearPredictionSelections();
            let classIdxs = [];
            if (!div.classList.contains("selected")) {
                classIdxs.push(parseInt(div.getAttribute("data-idx")));
                div.classList.add("selected");
            } else {
                div.classList.remove("selected");
            }

            for (const el of document.querySelectorAll("#grid div.selected")) {
                if (el !== div) {
                classIdxs.push(parseInt(el.getAttribute("data-idx")));
                }
            }

            if (classIdxs.length > 0) {
                this.clearSelection.disabled = false;
                $this.worker.postMessage({
                status: "class_by_heatmap",
                classIdxs: classIdxs,
                });
            } else {
                this.clearSelection.disabled = true;
                this.updateResults(
                this.results.heatmap,
                this.results.predictions,
                this.results.logits
                );
            }
        }
    });

    AppState.selectionEnabled = false;
  }

  _resizeCanvases(width, height) {
    this.hidden_canvas.width = width;
    this.hidden_canvas.height = height;

    this.img_canvas.width = this.min_side;
    this.img_canvas.height = this.min_side;

    this.heatmap_canvas.width = this.min_side;
    this.heatmap_canvas.height = this.min_side;
  }

  setSize(width, height) {
    this.width = width;
    this.height = height;
    this.min_side = Math.min(width, height);
    this.sx = Math.floor((width - this.min_side) / 2);
    this.sy = Math.floor((height - this.min_side) / 2);

    this._resizeCanvases(width, height);
  }

  getImage(img) {
    this.ctx_hidden.drawImage(img, 0, 0);
    return this.ctx_hidden.getImageData(0, 0, this.width, this.height);
  }

  load_selected_image(post_status) {
    const file = document.getElementById("predefined-files").value;
    if (!file) return;

    fetch("assets/" + file)
      .then((response) => response.blob())
      .then((blob) => this._handleUploadedBlob(blob, post_status))
      .catch((err) => console.warn("Failed to load asset:", err));
  }

  async _handleUploadedBlob(file, post_status) {
    this.video.pause();
    this.startButton.disabled = true;   

    if (this.video.srcObject) {
      this.video.srcObject.getTracks().forEach(t => t.stop());
      this.video.srcObject = null;
    }
    
    if (this.video.src) {
      this.video.removeAttribute('src');
      this.video.load();
    }

    if (file.type.startsWith("image/")) {
      const bitmap = await createImageBitmap(file);
      this.setSize(bitmap.width, bitmap.height);
      AppState.currentImage = bitmap;

      this._postMessage(post_status, bitmap);
    } else if (file.type.startsWith("video/")) {
      this.video.preload = "metadata";
      this.video.muted = true;
      this.video.playsInline = true;
      const url = URL.createObjectURL(file);
      this.video.src = url;
      this.video.load();

    } else {
      console.warn("Unsupported file type:", file.type);
      return;
    }
  }

  async initCamera() {
    let $this = this;

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      console.warn("Camera API not supported in this browser");
      return;
    }

    try {
      this.localMediaStream = await navigator.mediaDevices.getUserMedia({video: true});
      window._lafamMediaStream = this.localMediaStream;
    } catch (error) {
      console.warn("Camera permission denied or device not found", error);
      debug_devices.textContent +=
        "Camera permission denied or device not found\n";

      return;
    }

    let cameras = await navigator.mediaDevices.enumerateDevices();

    // Debug output
    if (typeof debug_devices !== "undefined") {
      debug_devices.textContent = JSON.stringify(cameras, null, 2);
    }

    cameras = cameras.filter((device) => device.kind === "videoinput");

    if (cameras.length === 0) {
      return;
    }
    console.log("Cameras found:", cameras);
    this.mainSection.classList.add("camera-available");

    const targetDeviceId = cameras[cameras.length - 1].deviceId;

    const currentTrack = this.localMediaStream.getVideoTracks()[0];
    const currentStreamId = currentTrack.getSettings().deviceId;
    this.currentCameraId = currentStreamId;

    if (currentStreamId !== targetDeviceId) {
      currentTrack.stop();

      this.localMediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          height: { ideal: 1024 },
        },
      });
      window._lafamMediaStream = this.localMediaStream;
    } else {
      this.currentCameraId = currentStreamId;
    }

    this.video.srcObject = this.localMediaStream;

    this.switchCameraButton.onclick = async () => {
      $this.video.pause();
      $this.mainSection.classList.add("paused");
      $this.startButton.disabled = true;
      
      if (!$this.video.srcObject && $this.video.currentSrc) {
        $this.video.removeAttribute('src');
        $this.video.load();
      }
      
      if ($this.localMediaStream) {
        $this.localMediaStream.getTracks().forEach((track) => track.stop());
      }

      const nextCamera = cameras.find(
        (camera) => camera.deviceId !== $this.currentCameraId
      );

      if (nextCamera) {
        $this.currentCameraId = nextCamera.deviceId;
      }

      try {
        $this.localMediaStream = await navigator.mediaDevices.getUserMedia({
          video: {
            deviceId: { exact: $this.currentCameraId },
            height: { ideal: 1024 },
          },
        });
        window._lafamMediaStream = $this.localMediaStream;

        $this.mainSection.classList.remove("paused");
        $this.video.srcObject = $this.localMediaStream;
        $this.toggleModeSelect(false);

      } catch (err) {
        console.error("Error switching camera:", err);
        addLogMsg("Error switching camera: " + err);
      }
    };

    // this.video.addEventListener("play", async () => {
    //   AppState.currentImage = await this._getFrame();
    //   this._postMessage(this.modeSelect.value, AppState.currentImage);
    // });
  }

  async _getFrame() {
    if (!this.video.srcObject) {
      if (typeof this.video.requestVideoFrameCallback === "function") {
        await new Promise((resolve) => this.video.requestVideoFrameCallback(resolve));
      }
    }
    
    return await createImageBitmap(
      this.video,
      this.sx, this.sy,
      this.min_side, this.min_side
    );
  }

  _onmessage(e) {
    const { data } = e;
    // Handle errors from worker
    if (data.status === "error") {
      document.getElementById("loading-indicator").style.display = "none";
      addLogMsg("Worker Error: " + data.message);
      alert("Failed to initialize: " + data.message);
      return;
    }
    
    if (data.status === "ready") {
      AppState.isWorkerReady = true;
      document.getElementById("loading-indicator").style.display = "none";
      this.mainSection.classList.add("ready");
      if(AppState.currentImage){
        this._postMessage();
      }
    }
    if (data.status === "results") {
      this.results = data;
      this._updateVideo();
      this.updateResults(data.heatmap, data.predictions, data.logits);
    }
    if (data.status === "weighted_heatmap") {
      this.updateHeatmap(data.heatmap);
    }
    if (data.status === "class_by_heatmap") {
      const top_n_idx = argmax_top_n(data.logits, TOP_N, 0);
      this._updatePredictionList(top_n_idx, data.predictions, data.logits);
    }

    if (data.status === "square_results_for_groupmap") {
      AppState.squareData = this.preprocessSquareResults(data.data);
      this.updateGroupMap();
      this.updateGroupList();
      this._updateVideo();
    }

    if (data.status === "square_results_for_classmap") {
      AppState.squareData = this.preprocessSquareResults(data.data);
      this.updateClassMap();
      this.updateClassList();
      this._updateVideo();
    }

    if (data.status === "embedding_result") {
        const obj = AppState.trackingObjects.find(o => o.id === data.id);
        if (obj) {
            obj.refEmb = data.refEmb;
            // Convert predictionIdx to class name
            const classInfo = this.imagenet_classes[data.predictionIdx];
            obj.name = `[${data.predictionIdx}] ${classInfo}`;
            this.renderTrackingObjectsList();
        }
    }

    if (data.status === "tracking_results") {
      this.heatmap_canvas.classList.add("hidden");
      this.drawTrackingResults(data.objects);
      this._updateVideo();
      // this.ctx_img.drawImage(AppState.currentImage, 0, 0, AppState.currentImage.width, AppState.currentImage.height);
    }
  }

  _clearPredictionSelections() {
    for (const el of document.querySelectorAll(".prediction.selected")) {
      el.classList.remove("selected");
    }
  }

  _clearHeatmapSelections() {
    // remove all selected classes and inline styles
    for (const el of document.querySelectorAll("#grid div.selected")) {
      el.classList.remove("selected");
      el.style.borderColor = "";
      el.style.boxShadow = "";
    }
  }

  _clearSelections() {
    this._clearPredictionSelections();
    this._clearHeatmapSelections();
    this.clearSelection.disabled = true;
  }

  async _postMessage() {
    this.toggleModeSelect(false);
    
    const activeObjects = AppState.trackingObjects.filter(o => o.cells.length > 0 && o.visible && o.refEmb);
    this.worker.postMessage({
                status: this.modeSelect.value,
                imageBitmap: AppState.currentImage,
                objects: activeObjects.map(o => ({
                    id: o.id,
                    refEmb: o.refEmb,
                    prevX: o.prevX || -1, // undefined becomes -1
                    prevY: o.prevY || -1
                })),
                settings: AppState.trackingSettings
            });
  }

  processModeChange() {
    const value = this.modeSelect.value;
    this.heatmap_canvas.classList.remove("hidden");
    this.trackingControls.classList.add("hidden");

    this._clearSelections();
    if (value === "predict") {
    } else if (value === "imagenet_classes") {
      this.showImagenetClasses();
    } else if (value === "groupmap_bbs") {
      this.showGroupmap();
    } else if (value === "object_tracking") {
        this.showObjectTracking();
    } else {
      console.error("Invalid mode-select value: ", value);
    }
    this._postMessage();
  }

  updateResults(heatmap, predictions, logits) {
    this._clearSelections();
    this._clearGroupMapDisplay();
    this._clearHeatmap();
    this._clearBoundingBoxes();
    this.toggleOpacitySlider(true);
    this.togglePaletteSelect(true);
    toggleSelection(true);

    this.ctx_img.drawImage(AppState.currentImage, 0, 0, this.min_side, this.min_side);
    this.updateHeatmap(heatmap);
    let top_n_idx = argmax_top_n(logits, TOP_N, 1.7);
    this._updatePredictionList(top_n_idx, predictions, logits);
  }

  async _updateVideo() {
    if (!this.video.paused) {
      AppState.currentImage = await this._getFrame();
      await this._postMessage();
      this.ctx_img.drawImage(AppState.currentImage, 0, 0, AppState.currentImage.width, AppState.currentImage.height);
    } else {
      this.toggleModeSelect(true);
    }
  }

  showGroupmap() {
    // if (AppState.currentImage === null) return;
    if (AppState.squareData === null) {
      // this._postMessage();
    } else {
      this.updateGroupMap();
      this.updateGroupList();
    }

    this._updateVideo();
  }

  showImagenetClasses() {
    // if (AppState.currentImage === null) return;
    if (AppState.squareData === null) {
      // this._postMessage();
    } else {
      this.updateClassMap();
      this.updateClassList();
    }

    this._updateVideo();
  }

  updateGroupMap() {
    this._clearSelections();
    this._clearGroupMapDisplay();
    this._clearHeatmap();
    this._clearBoundingBoxes();
    this.togglePaletteSelect(false);
    this.toggleModeSelect(true);
    toggleSelection(false);

    // show logits+classId in square
    AppState.squareData.forEach((square, i) => {
      const div = document.querySelector(`div[data-idx='${i}']`);
      if (div !== null) {
        const group = square.groupId;
        const logit = square.logit.toFixed(2);
        div.classList.add("class-display");
        div.innerHTML = `${logit}<br/>${group}`;
      } else {
        addLogMsg("Error: could not find div with data-idx=" + i);
      }
    });

    // calc and show groupmap
    const groupIds = AppState.squareData
      .map((square) => square.classId)
      .map((classId) => AppState.grouper.classToGroup(classId));
    const heatmap = this.makeClassHeatmap(groupIds);
    this.setClassHeatmap(heatmap);

    // calc and show group bounding boxes
    const groupGrid2d = to2DArray(
      AppState.squareData.map((square) => square.groupId),
      7,
      7
    );
    const areas = findAreas(groupGrid2d, [-1]);
    const boundingBoxes = findBoundingBoxes(areas);
    drawBoundingBoxes(boundingBoxes, "bounding-boxes-canvas");
  }

  updateGroupList() {
    // reuse prediction list div
    if (AppState.squareData === null) return;
    this._clearPredictionsList();
    let addedGroups = {};
    AppState.squareData
      .toSorted((a, b) => a.groupId - b.groupId)
      .filter((square) => square.groupId >= 0)
      .forEach((square) => {
        if (!addedGroups[square.groupId]) {
          const div = document.createElement("div");
          div.style.padding = "10px";
          div.style.fontWeight = "bold";
          div.innerText = `${square.groupId}: ${AppState.grouper.getGroupName(
            square.groupId
          )}`;
          this.predictionList.appendChild(div);
          addedGroups[square.groupId] = true;
        }
      });
  }

  preprocessSquareResults(data) {
    if (!AppState.grouper) {
      AppState.grouper = new ClassGrouper();
    }
    let squareData = [];
    for (let i = 0; i < 49; i++) {
      squareData.push({
        logit: parseFloat(data.logits[i]),
        classId: parseInt(data.classIds[i]),
        groupId: AppState.grouper.classToGroup(parseInt(data.classIds[i])),
      });
    }
    return squareData;
  }

  makeClassHeatmap(squareClasses) {
    const min = Math.min(...squareClasses);
    const max = Math.max(...squareClasses);
    return squareClasses.map((classId) => (classId - min) / (max - min + 1e-6));
  }

  _clearGroupMapDisplay() {
    for (let div of document.querySelectorAll("#grid div")) {
      div.innerHTML = "";
      div.classList.remove("class-display");
    }
  }

  _clearHeatmap() {
    const canvas = this.heatmap_canvas;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  _clearBoundingBoxes() {
    const canvas = this.bb_canvas;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  setClassHeatmap(data) {
    this.currentHeatmap = data;

    const hueBase = 20; // Math.floor(360 * Math.random());
    let heatmap = mapToHSL(data, hueBase);
    heatmap = new ImageProcessor(heatmap, 7, 7).resize(
      this.min_side,
      this.min_side,
      "nearest"
    );

    this.ctx_heatmap.putImageData(
      new ImageData(
        ImageProcessor.toImageData(heatmap),
        this.min_side,
        this.min_side
      ),
      0,
      0
    );
  }

  updateHeatmap(data, palette = null) {
    this.currentHeatmap = data;

    palette = palette === null ? this.currentPalette : palette;
    let heatmap = mapToPalette(data, this.palettes[palette]);
    heatmap = new ImageProcessor(heatmap, 7, 7).resize(
      this.min_side,
      this.min_side,
      "nearest"
    );

    this.ctx_heatmap.putImageData(
      new ImageData(
        ImageProcessor.toImageData(heatmap),
        this.min_side,
        this.min_side
      ),
      0,
      0
    );
  }

  _updatePredictionList(indices, predictions, logits) {
    // create list with progress bars
    this.predictionList.innerHTML = "";
    const fragment = document.createDocumentFragment();
    for (let i = 0; i < indices.length; i++) {
      const idx = indices[i];
      let div = document.createElement("div");
      div.classList.add("prediction");
      div.setAttribute("data-idx", idx);
      // div.setAttribute("data-logits", logits[idx]);

      let label = document.createElement("label");
      const cls = this.imagenet_classes[idx];
      const prob = predictions[idx];
      const l = logits[idx].toFixed(2);
      label.innerHTML = `<b>${cls}</b> (Logits: ${l}; Softmax: ${Math.round(
        prob * 100
      )}%)`;
      div.appendChild(label);

      let progress = document.createElement("progress");
      progress.value = prob;
      progress.max = 1;
      div.appendChild(progress);

      fragment.appendChild(div);
    }

    this.predictionList.appendChild(fragment);
  }

  toggleOpacitySlider(enabled, value = null) {
    if (value !== null) {
      this.heatmapOpacity.value = value;
      this.heatmap_canvas.style.opacity = this.heatmapOpacity.value;
    }
    this.heatmapOpacity.disabled = !enabled;
  }

  toggleModeSelect(enabled, value = null) {
    if (value !== null) {
      this.modeSelect.value = value;
    }
    this.modeSelect.disabled = !enabled;
  }

  togglePaletteSelect(enabled, value = null) {
    if (value !== null) {
      this.paletteSelect.value = value;
      this.currentPalette = this.paletteSelect.value;
      this.updateHeatmap(this.currentHeatmap);
    }
    this.paletteSelect.disabled = !enabled;
  }

  _clearPredictionsList() {
    this.predictionList.innerHTML = "";
  }

  updateClassMap() {
    this._clearSelections();
    this._clearGroupMapDisplay();
    this._clearHeatmap();
    this._clearBoundingBoxes();
    this.togglePaletteSelect(false);
    this.toggleModeSelect(true);
    toggleSelection(false);

    // show logits+classId in square
    AppState.squareData.forEach((square, i) => {
      const div = document.querySelector(`div[data-idx='${i}']`);
      if (div !== null) {
        const classId = square.classId;
        const logit = square.logit.toFixed(2);
        div.classList.add("class-display");
        div.innerHTML = `${logit}<br/>${classId}`;
      } else {
        addLogMsg("Error: could not find div with data-idx=" + i);
      }
    });

    // calc and show group map
    const classIds = AppState.squareData.map((square) => square.classId);
    const heatmap = this.makeClassHeatmap(classIds);
    this.setClassHeatmap(heatmap);
  }

  updateClassList() {
    if (AppState.squareData === null) return;
    this._clearPredictionsList();
    const seen = {};
    AppState.squareData
      .map((square) => square.classId)
      .filter((classId, index, self) => {
        if (classId <= 0) return false;
        if (seen[classId]) return false;
        seen[classId] = true;
        return true;
      })
      .toSorted((a, b) => a - b)
      // .slice(0, 17)
      .forEach((classId) => {
        const div = document.createElement("div");
        div.style.padding = "10px";
        div.style.fontWeight = "bold";
        div.innerText = `${classId}: ${this.imagenet_classes[classId]}`;
        this.predictionList.appendChild(div);
      });
  }

  showObjectTracking() {
    // Show tracking controls
    this.trackingControls.classList.remove("hidden");
    
    // Hide standard prediction list/sliders if desired, or repurpose them
    // For now we keep them but maybe standard predictions are less relevant unless looking at a paused frame
    
    // Clear other views
    this._clearGroupMapDisplay();
    this._clearBoundingBoxes();
    
    // Enable selection (for defining objects)
    AppState.selectionEnabled = true;
    toggleSelection(true);

    if (AppState.currentImage === null) {
        //  this._postMessage();
         return; 
    }
    
    // Redraw grid for active object
    if (this.video.paused) {
        this._clearSelections();
        // Highlight active object cells
        const obj = AppState.trackingObjects.find(o => o.id === AppState.activeTrackingObjectId);
        if (obj) {
            for(let idx of obj.cells) {
                const div = document.querySelector(`div[data-idx='${idx}']`);
                if(div) {
                    div.classList.add("selected");
                    div.style.borderColor = obj.color;
                    div.style.boxShadow = `0 0 20px ${obj.color} inset`;
                }
            }
             // Trigger prediction view for these cells
             if (obj.cells.length > 0) {
                 this.worker.postMessage({
                    status: "class_by_heatmap",
                    classIdxs: obj.cells,
                 });
             }
        }
    }
    
    this._updateVideo();
  }

  addTrackingObject() {
      const id = Date.now();
      const colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF", "#FF00FF"];
      const color = colors[AppState.trackingObjects.length % colors.length];
      
      const newObj = {
          id: id,
          name: `Object ${AppState.trackingObjects.length + 1}`,
          color: color,
          cells: [],
          refEmb: null,
          visible: true,
          prevX: null,
          prevY: null
      };
      
      AppState.trackingObjects.push(newObj);
      this.selectTrackingObject(id);
      this.renderTrackingObjectsList();
  }

  removeTrackingObject(id) {
      AppState.trackingObjects = AppState.trackingObjects.filter(o => o.id !== id);
      if (AppState.activeTrackingObjectId === id) {
          AppState.activeTrackingObjectId = AppState.trackingObjects.length > 0 ? AppState.trackingObjects[0].id : null;
      }
      this.renderTrackingObjectsList();
      if(AppState.activeTrackingObjectId) this.selectTrackingObject(AppState.activeTrackingObjectId);
  }

  selectTrackingObject(id) {
      AppState.activeTrackingObjectId = id;
      this.renderTrackingObjectsList();
      
      // Refresh grid selection if paused
      if (this.video.paused) {
          this._clearSelections();
          const obj = AppState.trackingObjects.find(o => o.id === id);
          if (obj) {
              for(let idx of obj.cells) {
                  const div = document.querySelector(`div[data-idx='${idx}']`);
                  if(div) {
                      div.classList.add("selected");
                      div.style.borderColor = obj.color;
                      div.style.boxShadow = `0 0 20px ${obj.color} inset`;
                  }
              }
               // Trigger prediction view for these cells
             if (obj.cells.length > 0) {
                 this.worker.postMessage({
                    status: "class_by_heatmap",
                    classIdxs: obj.cells,
                 });
             } else {
                 this.updateResults(
                    this.results.heatmap, // Use last available heatmap? Or clear
                    this.results.predictions,
                    this.results.logits
                 );
             }
          }
      }
  }

  renderTrackingObjectsList() {
      this.trackingObjectsList.innerHTML = "";
      
      AppState.trackingObjects.forEach(obj => {
          const div = document.createElement("div");
          div.className = "prediction"; // reuse class for styling
          if (obj.id === AppState.activeTrackingObjectId) {
              div.classList.add("selected");
          }
          
          div.style.borderLeft = `5px solid ${obj.color}`;
          
          const info = document.createElement("span");
          info.innerText = obj.name;
          info.style.flex = "1";
          info.onclick = () => this.selectTrackingObject(obj.id);
          
          const removeBtn = document.createElement("button");
          removeBtn.innerHTML = "X"; // or icon
          removeBtn.style.background = "none";
          removeBtn.style.border = "none";
          removeBtn.style.cursor = "pointer";
          removeBtn.style.color = "var(--text-muted)";
          removeBtn.onclick = (e) => {
              e.stopPropagation();
              this.removeTrackingObject(obj.id);
          };

          div.appendChild(info);
          div.appendChild(removeBtn);
          
          this.trackingObjectsList.appendChild(div);
      });
  }

  drawTrackingResults(results) {
        // Draw standard video output first (clears previous frame)
        // If we want to keep the heatmap from tracking, we'd need to blend it.
        // For now, let's just draw the tracked cross on the bounding box canvas
        this._clearBoundingBoxes();
        
        const ctx = this.bb_canvas.getContext("2d");
        const scale = this.min_side; // Since coordinates are normalized to [0,1] or 224 based on worker?
        // Worker returns pixel coordinates in input space (224x224) or relative?
        // Let's assume Worker returns normalized [0..1] to be resolution independent?
        // OR Worker knows input size 224. 
        // Plan says: "Upsample... Return list of { id, x, y }" where x,y are in 224 space or original?
        // Worker.js has INPUT_WIDTH = 224.
        
        // Let's assume x,y are in [0, 224]. We scale to min_side.
        const r = this.min_side / 224;

        // Optionally clear heatmap or specific overlay
        // Implementation Plan says: "Overlay Alpha"
        if (AppState.trackingSettings.overlayAlpha < 1.0) {
              // ...
        }

        results.forEach(res => {
            const obj = AppState.trackingObjects.find(o => o.id === res.id);
            if (!obj) return;
            
            // Update prev pos
            obj.prevX = res.x;
            obj.prevY = res.y;
            
            if (res.x < 0 || res.y < 0) return; // Not found / below threshold

            const x = res.x * r;
            const y = res.y * r;
            const CrossSize = this.min_side * 0.05;

            ctx.beginPath();
            ctx.strokeStyle = obj.color;
            ctx.lineWidth = 3;
            // Draw cross
            ctx.moveTo(x - CrossSize, y);
            ctx.lineTo(x + CrossSize, y);
            ctx.moveTo(x, y - CrossSize);
            ctx.lineTo(x, y + CrossSize);
            ctx.stroke();
            
            // Draw Label
            ctx.font = "16px sans-serif";
            ctx.fillStyle = obj.color;
            ctx.fillText(obj.name, x + 5, y - 5);
        });
  }
}


function toggleSelection(enabled) {
  for (let div of document.querySelectorAll("#grid div")) {
    if (enabled) {
      div.classList.add("grid-selection");
    } else {
      div.classList.remove("grid-selection");
    }
  }
  AppState.selectionEnabled = enabled;
}

async function init() {
  if ("serviceWorker" in navigator) {
    try {
      await navigator.serviceWorker.register("service-worker.js");
    } catch (error) {
      console.error("Service Worker Registration Failed:", error);
    }
  }

  new ModelWorker("js/worker.js");
}

function mapToHSL(x, hueBase = 0) {
  let heatmap = [
    new Float32Array(x.length),
    new Float32Array(x.length),
    new Float32Array(x.length),
  ];
  for (let i = 0; i < x.length; i++) {
    const hue = (hueBase + 300 * x[i]) % 360; // 360 * x[i]; // degrees
    const saturation = 100; // %
    const lightness = 50; // %
    const [r, g, b] = hslToRgb(hue, saturation, lightness);

    heatmap[0][i] = r;
    heatmap[1][i] = g;
    heatmap[2][i] = b;
  }
  return heatmap;
}

function mapToPalette(x, palette) {
  let heatmap = [
    new Float32Array(x.length),
    new Float32Array(x.length),
    new Float32Array(x.length),
  ];
  for (let i = 0; i < x.length; i++) {
    const color = palette[Math.round(x[i] * (palette.length - 1))];
    heatmap[0][i] = color[0];
    heatmap[1][i] = color[1];
    heatmap[2][i] = color[2];
  }
  return heatmap;
}

function hslToRgb(h, s, l) {
  // Normalize the H, S, L values
  s /= 100;
  l /= 100;

  const C = (1 - Math.abs(2 * l - 1)) * s; // Chroma
  const X = C * (1 - Math.abs(((h / 60) % 2) - 1)); // Secondary component
  const m = l - C / 2; // Adjustment factor

  let rPrime, gPrime, bPrime;

  // Determine RGB prime values based on the hue angle
  if (h >= 0 && h < 60) {
    rPrime = C;
    gPrime = X;
    bPrime = 0;
  } else if (h >= 60 && h < 120) {
    rPrime = X;
    gPrime = C;
    bPrime = 0;
  } else if (h >= 120 && h < 180) {
    rPrime = 0;
    gPrime = C;
    bPrime = X;
  } else if (h >= 180 && h < 240) {
    rPrime = 0;
    gPrime = X;
    bPrime = C;
  } else if (h >= 240 && h < 300) {
    rPrime = X;
    gPrime = 0;
    bPrime = C;
  } else {
    rPrime = C;
    gPrime = 0;
    bPrime = X;
  }

  // Convert RGB prime values to RGB values and adjust by m
  const r = Math.round((rPrime + m) * 255);
  const g = Math.round((gPrime + m) * 255);
  const b = Math.round((bPrime + m) * 255);

  return [r, g, b];
}

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

    for (let channel = 0; channel < this.channels.length; channel++) {
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

class ClassGrouper {
  constructor() {
    this.includeGroups = AppState.includeGroups;
    this.excludeGroups = AppState.excludeGroups;
    this.includedClasses = new Set(Object.values(this.includeGroups).flat());
    this.excludedClasses = new Set(Object.values(this.excludeGroups).flat());

    // mapping for group names
    this._classToGroup = {};
    for (let [groupName, indices] of Object.entries(this.includeGroups)) {
      for (let idx of indices) {
        this._classToGroup[idx] = groupName;
      }
    }

    this._groupToId = {};
    Object.entries(this.includeGroups).forEach(([group, _], groupId) => {
      this._groupToId[group] = groupId;
    });
  }

  classToGroup(classId) {
    if (this.excludedClasses.has(classId)) return -1;
    const group = this._classToGroup[classId];
    const groupId = this.getGroupId(group);
    if (groupId < 0 || groupId === undefined) return -1;
    return groupId;
  }

  getGroupId(group) {
    return this._groupToId[group];
  }

  getGroupName(groupId) {
    if (groupId < 0) return "---";
    return Array.from(Object.keys(this.includeGroups))[groupId];
  }
}

function argmax_top_n(logits, n, threshold = 0) {
  let indices = logits.map((e, i) => i);
  indices.sort((a, b) => logits[b] - logits[a]);
  let top_n = [];
  for (let i = 0; i < n; i++) {
    if (logits[indices[i]] < threshold) break;

    top_n.push(indices[i]);
  }
  return top_n; //.slice(0, n);
}

function updateServiceWorker() {
  if ("caches" in window) {
    caches
      .keys()
      .then((keyList) => {
        return Promise.all(
          keyList.map((key) => {
            return caches.delete(key);
          })
        );
      })
      .then(() => {
        console.log("All caches cleared.");
      })
      .catch((err) => {
        console.error("Error clearing caches:", err);
      });
  }

  if ("serviceWorker" in navigator) {
    navigator.serviceWorker
      .getRegistrations()
      .then((registrations) => {
        // Unregister all service workers
        for (let registration of registrations) {
          registration.unregister();
        }
      })
      .catch((error) => {
        console.error("Error unregistering service worker:", error);
      });
  }
  localStorage.removeItem('lafam_tutorial_shown');
  window.location.reload();
}

let loadingText = document.getElementById("loading-text");

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.addEventListener("message", (event) => {
    if (event.data && event.data.type === "progress") {
      loadingText.textContent = `Downloading ${event.data.name} (${event.data.progress}%)`;
    }
  });
}

// release camera on page unload
window.addEventListener("beforeunload", () => {
  if (window._lafamMediaStream) {
    window._lafamMediaStream.getTracks().forEach((track) => track.stop());
  }
});

const grid = document.getElementById("grid");
const cellCount = 49; // 7 x 7 grid for example

for (let i = 0; i < cellCount; i++) {
  const cell = document.createElement("div");
  cell.dataset.idx = i;
  grid.appendChild(cell);
}

init();
