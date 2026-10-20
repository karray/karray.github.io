const IMAGENET_CLASSES = "/demos/models/imagenet_classes.json";
const TISSUE_CLASSES = "/demos/models/tissue_classes.json";
const TUMOUR_CLASSES = "/demos/models/tumour_classes.json";
const MODEL_FILES = "https://huggingface.co/karay/demo-models/resolve/9ae07a86944dce671bd8d4135081f2b92e0e53d9/";
const MODEL_FILES_2 = "https://huggingface.co/karay/demo-models/resolve/14b15a321eca307e50770bb09df1d53460b4d30b/";   // the pathology models and CLIP RN50x4
const H_AND_E = "H&E at about 0.5 µm per pixel (20×): 224 pixels = 112 µm";

const MODEL_GROUPS = {
  classifier: "Classifiers (ImageNet)",
  ssl: "Self-supervised",
  clip: "Image-text (CLIP)",
  pathology: "Pathology (H&E)",
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
  { id: "clip_rn50x4", label: "CLIP RN50x4 (OpenAI), zero-shot ImageNet", group: "clip", url: MODEL_FILES_2 + "clip_rn50x4.onnx", mb: 171, size: 288, classes: IMAGENET_CLASSES, pooling: "attention" },

  { id: "hibou_b", label: "Hibou-B + tissue head", group: "pathology", url: MODEL_FILES_2 + "hibou_b.onnx", mb: 164, size: 224, classes: TISSUE_CLASSES, scale: H_AND_E },
  { id: "quiltnet_b32", label: "QuiltNet-B-32 (pathology CLIP), zero-shot tissue", group: "pathology", url: MODEL_FILES_2 + "quiltnet_b32.onnx", mb: 168, size: 224, classes: TISSUE_CLASSES, pooling: "token", scale: H_AND_E },
  { id: "tumour_detector", label: "Breast tumour detector (ResNet-34)", group: "pathology", url: MODEL_FILES_2 + "tumour_detector.onnx", mb: 41, size: 224, classes: TUMOUR_CLASSES, scale: "H&E at about 0.4 µm per pixel: 224 pixels = 87.5 µm" },
  { id: "kaiko_vit_small", label: "Kaiko ViT-S/16", group: "pathology", url: MODEL_FILES_2 + "kaiko_vit_small.onnx", mb: 42, size: 224, classes: null, scale: H_AND_E },
  { id: "ctranspath", label: "CTransPath (Swin-T)", group: "pathology", url: MODEL_FILES_2 + "ctranspath.onnx", mb: 55, size: 224, classes: null, scale: H_AND_E },
];
