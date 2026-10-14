// The models shared by the demos, in the order of their model menus. A model is downloaded only when
// it is selected. The files are on Hugging Face (https://huggingface.co/karay/demo-models), pinned to
// a commit, so that a changed file gets a new URL and is never served stale from a cache. To add a
// model, export it (tmp/models/export_models.py, not part of the site), upload it to that repo and
// add an entry here with the new commit.
//   inputs: image (RGB in [0, 1], size x size); outputs: features and, for classifiers, cell_logits
//   classes: the class names of a classifier, null for a model without a classifier
const IMAGENET_CLASSES = "/demos/models/imagenet_classes.json";
const MODEL_FILES = "https://huggingface.co/karay/demo-models/resolve/9ae07a86944dce671bd8d4135081f2b92e0e53d9/";

const MODEL_GROUPS = {
  classifier: "Classifiers (ImageNet)",
  ssl: "Self-supervised",
  clip: "Image-text (CLIP)",
};

const MODELS = [
  { id: "resnet50", label: "ResNet-50", group: "classifier", url: MODEL_FILES + "resnet50.onnx", mb: 49, size: 224, classes: IMAGENET_CLASSES },
  { id: "convnext_small", label: "ConvNeXt-S", group: "classifier", url: MODEL_FILES + "convnext_small.onnx", mb: 96, size: 224, classes: IMAGENET_CLASSES },
  { id: "swin_small", label: "Swin-S", group: "classifier", url: MODEL_FILES + "swin_small.onnx", mb: 98, size: 224, classes: IMAGENET_CLASSES },
  { id: "maxvit_tiny", label: "MaxViT-T", group: "classifier", url: MODEL_FILES + "maxvit_tiny.onnx", mb: 60, size: 224, classes: IMAGENET_CLASSES },

  { id: "dino_resnet50", label: "DINO ResNet-50", group: "ssl", url: MODEL_FILES + "dino_resnet50.onnx", mb: 45, size: 224, classes: null },
  { id: "dino_vit_small", label: "DINO ViT-S/16", group: "ssl", url: MODEL_FILES + "dino_vit_small.onnx", mb: 42, size: 224, classes: null },
  { id: "dinov2_vit_small", label: "DINOv2 ViT-S/14", group: "ssl", url: MODEL_FILES + "dinov2_vit_small.onnx", mb: 41, size: 224, classes: null },
  { id: "dinov2_reg_vit_small", label: "DINOv2 ViT-S/14 with registers", group: "ssl", url: MODEL_FILES + "dinov2_reg_vit_small.onnx", mb: 41, size: 224, classes: null },
  { id: "dinov3_vit_small_plus", label: "DINOv3 ViT-S+/16", group: "ssl", url: MODEL_FILES + "dinov3_vit_small_plus.onnx", mb: 55, size: 224, classes: null },
  { id: "dinov3_convnext_small", label: "DINOv3 ConvNeXt-S", group: "ssl", url: MODEL_FILES + "dinov3_convnext_small.onnx", mb: 95, size: 224, classes: null },
  { id: "fcmae_convnextv2_tiny", label: "ConvNeXt V2-T (masked autoencoder)", group: "ssl", url: MODEL_FILES + "fcmae_convnextv2_tiny.onnx", mb: 53, size: 224, classes: null },

  { id: "clip_resnet101", label: "CLIP ResNet-101 (OpenAI)", group: "clip", url: MODEL_FILES + "clip_resnet101.onnx", mb: 81, size: 224, classes: null },
  { id: "tinyclip_vit_medium", label: "TinyCLIP ViT-M/16", group: "clip", url: MODEL_FILES + "tinyclip_vit_medium.onnx", mb: 73, size: 224, classes: null },
];
