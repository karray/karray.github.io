// The models shared by the demos, in the order of their model menus. A model is downloaded only when
// it is selected. The files are on Hugging Face (https://huggingface.co/karay/demo-models), pinned to
// a commit, so that a changed file gets a new URL and is never served stale from a cache.
const IMAGENET_CLASSES = "/examples/models/imagenet_classes.json";
const MODEL_FILES = "https://huggingface.co/karay/demo-models/resolve/9ae07a86944dce671bd8d4135081f2b92e0e53d9/";

const MODELS = [
  // inputs: image (RGB in [0, 1], size x size); outputs: features, cell_logits
  { id: "resnet50", label: "ResNet-50", url: MODEL_FILES + "resnet50.onnx", mb: 49, size: 224, classes: IMAGENET_CLASSES },
  { id: "convnext_small", label: "ConvNeXt-S", url: MODEL_FILES + "convnext_small.onnx", mb: 96, size: 224, classes: IMAGENET_CLASSES },
  { id: "swin_small", label: "Swin-S", url: MODEL_FILES + "swin_small.onnx", mb: 98, size: 224, classes: IMAGENET_CLASSES },
  { id: "maxvit_tiny", label: "MaxViT-T", url: MODEL_FILES + "maxvit_tiny.onnx", mb: 60, size: 224, classes: IMAGENET_CLASSES },
];
