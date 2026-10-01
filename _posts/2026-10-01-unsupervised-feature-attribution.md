---
layout: post
styles:
  - /assets/css/lafam.css
  - /assets/css/posts/lafam-header.css
header: lafam.html
header_class: lafam-header
title: What Does a Vision Model See Without Labels?
date: 2026-10-01 11:40:00 +0200
tags: XAI, CNN, ViT, Foundation Models, Self-Supervised Learning
description: How activation maps and relevance propagation can shed light on deep models without any labels.
image:
  path: /assets/img/posts/lafam/preview.png
  width: 1200
  height: 630
  alt: Two images beside heatmaps for a MaxViT encoder, produced by averaging, AnyUp, and LRP.

lafam_results:
  kicker: ConvNeXt Base · Averaging
  opacity: 100
  columns: 5
  items:
    - label: Dog and cat
      src: /assets/img/posts/lafam/figures/images/dog_cat.jpeg
      overlay: /assets/img/posts/lafam/figures/heatmaps/lafam_ConvNeXtBaseIN1K_dog_cat.png
    - label: Two dogs
      src: /assets/img/posts/lafam/figures/images/dog_dog.jpeg
      overlay: /assets/img/posts/lafam/figures/heatmaps/lafam_ConvNeXtBaseIN1K_dog_dog.png
    - label: Fish and person
      src: /assets/img/posts/lafam/figures/images/fish_person.png
      overlay: /assets/img/posts/lafam/figures/heatmaps/lafam_ConvNeXtBaseIN1K_fish_person.png
    - label: Spider web
      src: /assets/img/posts/lafam/figures/images/spider_web.png
      overlay: /assets/img/posts/lafam/figures/heatmaps/lafam_ConvNeXtBaseIN1K_spider_web.png
    - label: Six objects
      src: /assets/img/posts/lafam/figures/images/misc.png
      overlay: /assets/img/posts/lafam/figures/heatmaps/lafam_ConvNeXtBaseIN1K_misc.png
---

Foundation models trained without explicit labels can learn remarkably rich representations, but the same lack of supervision that makes this possible also makes them harder to interpret and validate. Without annotations to tell us what the model has learned, where can we look for clues? Convolutional networks and vision transformers offer one in their activations. These activations can be turned into heatmaps that show where the model responds most strongly, without extra labels or gradients. The catch is that these maps are often coarse. In this blog post, we explore how to extract spatial heatmaps from these activations and how they can be refined.

<!--more-->

This journey started with my attempt to use generative adversarial networks (GANs) as feature extractors for medical images. GANs can synthesize high-quality images from random noise and provide control over the generation via latent variables, but they are not inherently invertible. I tried to address this limitation, but the approach did not work well (see my earlier [post](/2023/01/06/turning-stylegan-into-a-latent-feature-extractor.html) for details). Debugging that encoder-based GAN quickly became an exercise in frustration. More importantly, it exposed how few tools exist for debugging models trained without explicit annotations. The limitation is particularly relevant in self-supervised learning (SSL), where useful representations are learned directly from unlabeled data. This gap motivated me to investigate ways of deriving attribution maps for models trained without explicit labels. As part of this investigation, I found that simply averaging the feature activations produces surprisingly coherent spatial maps [<a href="#karjauv2024lafam" data-ref="karjauv2024lafam">Karjauv et al.</a>].

All figures and numbers in this post can be reproduced with the [companion notebook](https://github.com/karray/deep-unboxing/blob/main/01_label_free_maps.ipynb), which also contains the code for the maps and for LRP, as well as small step-by-step implementations of RISE, RELAX, CAM and Grad-CAM.

## Self-Supervised Learning Meets XAI

Learning without labels matters because collecting data is often much easier than annotating it, especially at scale. SSL makes it possible to use that unlabeled data to learn representations that can later be reused or fine-tuned for downstream tasks, such as image classification or segmentation. At the core of these models is an encoder, a network that turns an input image into a compact set of features called an embedding. In a supervised classifier, this embedding is typically passed to a prediction head that produces class scores. SSL models such as SimCLR and DINO instead train the encoder on a proxy task, such as matching different augmented views of the same image. The proxy task is only a means to learn good features. Once training is done, it is discarded, and what remains is an encoder with no task of its own. At sufficiently large scale, models trained this way can also form the basis of foundation models.

The absence of labels, however, also makes it difficult to verify whether the learned representations are actually relevant for a given downstream task. For example, a model trained on images of cars may learn features related to wheels, windows, and overall vehicle shape. Some of these features may transfer well to other types of vehicles, while others may be of little use when the model is applied to bicycles or motorcycles. Without labels, it is difficult to know which properties the model has learned and how useful they will be beyond the training data. Moreover, training on random crops can lead a model to memorize unintended associations, such as which object belongs in front of a particular background. Such associations can easily go unnoticed, and evaluating them is non-trivial [<a href="#meehan2023do" data-ref="meehan2023do">Meehan et al.</a>].

This is where explainable AI (XAI) would seem useful. For supervised models, methods such as Grad-CAM can highlight image regions that contribute to a particular prediction. Occlusion methods such as RISE take a different route. They repeatedly mask parts of the input and observe how the target prediction score changes.

For an SSL encoder, however, there may be no prediction score to explain. The model produces an embedding, and its individual dimensions usually have no predefined semantic meaning. We do not know that one feature represents a wheel, another a window, or another the shape of a car. Asking for the importance of a pixel for the class “car” therefore makes little sense when there is no car score in the first place.

There have been attempts to adapt XAI to label-free models. One such attempt is RELAX (Representation Learning Explainability), which extends the supervised occlusion method RISE. Instead of asking how masking part of an image changes a class score, we can ask how much the masking changes the representation. The original image provides a reference embedding. Masked versions of the image are passed through the same encoder, and their embeddings are compared with the reference using cosine similarity. If masking a region substantially changes the representation, that region was probably important to the encoder.

The idea is simple, but it comes at a cost. RELAX requires many masked versions of each image and therefore many forward passes through the model, while the resulting maps can be noisy. More generally, masking itself is not entirely innocent. Hiding parts of an image with a solid color creates inputs that differ from what the model normally sees, so part of the response may come from the perturbation.

### From Class-Specific to Label-Free Activation Maps (LaFAM)

One useful property of CNNs is that their convolutional feature maps retain a spatial layout. If a feature responds strongly in one part of the image, we can trace that response back to roughly the same region in the input. This spatial structure is what makes methods such as class activation maps (CAMs) possible.

<details><summary>What are activation maps and receptive fields?</summary>
<p>
Each convolutional layer produces a collection of activation maps (also called feature maps), one per feature channel. Each map records how strongly a learned feature responds at different spatial locations. Together, these maps form a feature tensor. Early layers often respond to relatively simple patterns such as edges or color contrasts. Deeper layers combine these responses into more complex representations. These can become selective to textures, shapes, object parts, or other structures useful to the model, although individual channels do not necessarily correspond to clean human-interpretable concepts.
</p>

<p>
Each activation is influenced by a region of the original image called its receptive field. Receptive fields grow as convolutional layers are stacked. Pooling and strided convolutions increase them further while reducing spatial resolution. As a result, deeper feature maps capture information from larger regions of the image but provide a progressively coarser spatial view.
</p>
</details>

In the original CAM formulation, the classifier assigns a different weight to each feature channel for each class. To explain a prediction such as *car*, CAM reduces the stack of spatial activation maps to a single heatmap using the weights associated with the car class. Features that are strongly associated with that class contribute more to the resulting heatmap. Grad-CAM generalizes this idea by using gradients to estimate how important each feature map is for a chosen target. In other words, the activation maps tell us where a feature responds, while the class-specific weights tell us which features matter for the target.

But what if there is no class to explain? We can take a surprisingly simple approach. Instead of finding class-specific weights, we give every feature channel equal weight and average the activations across channels at each spatial location. In the ResNet-50 models evaluated in the LaFAM paper, the final activations come out of a ReLU, which sets negative responses to zero. Positive and negative values therefore cannot cancel when averaged. A location receives a high value when many high-level features respond there, or when a smaller number respond particularly strongly. While CAM asks the classifier which feature maps matter and then combines them accordingly, averaging simply lets every feature map vote equally.

### What About Vision Transformers?

ViTs come with a tempting visualization: attention. We can inspect how image patches attend to one another and map those interactions back onto the image. In self-supervised models such as DINO, attention maps can even reveal surprisingly clear object structure without segmentation labels. But attention tells us how tokens exchange information, not necessarily which parts of the image are responsible for the representation we want to inspect. So what happens if we leave attention aside and look at the token representations themselves?

A ViT divides an image into patches and represents each patch with a token. The tokens are processed as a sequence and exchange information through self-attention, so a token in a deep layer no longer describes only its own patch. Still, every patch token has a known spatial address in the original image, which means we can arrange them back into their original grid. Many ViTs, such as the supervised ViT and CLIP, are trained through an extra class token, and only that token reaches the classifier. Even so, the patch tokens preserve where things are in the image [<a href="#raghu2021do" data-ref="raghu2021do">Raghu et al.</a>], and we can read this information from them. This gives us something similar to a CNN feature tensor. In a CNN, each spatial location contains a vector of channel activations. In a ViT, each patch contains a token vector, and we call its entries channels as well. Reduce each token to a single value, put the values back into the patch grid, and we get a coarse heatmap.

The reduction itself requires a choice. Unlike the ReLU outputs of a ResNet, ViT tokens are signed, and so are the features of modern CNNs such as ConvNeXt. Individual channels also have no predefined semantic meaning. Different features may respond to objects, object parts, textures, or surrounding context. Collapsing them into a single value therefore produces a summary of where the representation is active. Here, we apply ReLU before averaging, which keeps the positive responses and prevents them from cancelling against negative ones. We use this same reduction for every model below, so the maps show where positive feature responses are strongest. It is not the only reasonable choice, and it does not work equally well for every encoder, as we will see further below.

### Results

{% include heatmap-grid.html id="lafam-results" data=page.lafam_results %}

The grid shows label-free heatmaps from a supervised ConvNeXt for five images, four of which contain several objects. When a scene contains more than one object, the map does not pick a winner. The dog and the cat both light up, and so do both dogs and all six objects. With no class to choose, every object the encoder responds to strongly stays in the map.

The limitation is just as easy to see. ConvNeXt, like ResNet-50, ends with a $7\times7$ grid of features. The response is already informative, but it is coarse. Nearest-neighbor resizing shows it as large tiles, and bilinear resizing would only blend their edges.


<details markdown="1">
<summary>Supporting evidence: the evaluation in the LaFAM paper</summary>

The paper systematically compares averaging with RELAX (the prior SSL method) using SSL models (SimCLR and SwAV with ResNet-50 backbones). It also compares averaging with Grad-CAM for a fully supervised ResNet-50 classifier as a sanity check. The heatmaps for ResNet-50 are only $7\times7$ and were upscaled to the input image size with nearest-neighbor interpolation. Since we don't have class labels to evaluate the “correctness” of an explanation, the evaluation was performed on datasets with segmentation masks (ImageNet-S and PASCAL VOC), using a suite of metrics from the Quantus XAI evaluation framework. Essentially, we treat it as a localization task, with the goal of assessing how well the heatmaps align with the segmentation masks.

<figure>
<img src="/assets/img/posts/lafam/pascal_voc_2012_results.svg" alt="LaFAM vs. Grad-CAM vs. RELAX" />
  <figcaption>Comparison on PASCAL VOC 2012. Averaging produces heatmaps similar to Grad-CAM but remains robust when the model misclassifies (first row).</figcaption>
</figure>

The figure above compares averaging, RELAX, and Grad-CAM as a baseline. When applied to the same supervised model, averaging and Grad-CAM produce very similar heatmaps.

In the first row, the model predicted the wrong ImageNet class, so Grad-CAM highlighted a region associated with that wrong class. Averaging does not depend on the predicted label, so it still shows what the model found salient. For SSL models, averaging produces notably less noisy heatmaps than RELAX.

The next figure demonstrates that averaging, being class-agnostic, highlights multiple objects in the image (e.g., both dogs), whereas Grad-CAM focuses on the one tied to the predicted class.

![LaFAM for Multiple Objects](/assets/img/posts/lafam/pascal_voc_2012_2_classes.svg){: .center-image }

The last figure shows more misclassified ImageNet examples, with Grad-CAM computed for the wrong predicted class.

![Grad-CAM for Misclassification Case](/assets/img/posts/lafam/imgnet_missclf_short.svg){: .center-image }

The averaged heatmaps clearly highlight true objects, suggesting that these objects strongly activate multiple channels in the final convolutional layer. However, the prediction is wrong. The reason could be that the last fully connected layer puts more weight on specific features that mislead the final decision, even though many other features correctly identify the object.

<table id="imagenet_results">
  <caption>Heatmap performance comparison on ImageNet-S (higher values are better).</caption>
  <thead>
    <tr>
      <th rowspan="2">Metric</th>
      <th colspan="2">Supervised (ResNet-50)</th>
      <th colspan="2">SSL (SimCLR)</th>
      <th colspan="2">SSL (SwAV)</th>
    </tr>
    <tr>
      <th>Grad-CAM</th>
      <th>LaFAM</th>
      <th>RELAX</th>
      <th>LaFAM</th>
      <th>RELAX</th>
      <th>LaFAM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Pointing Game</td>
      <td><strong>94.00</strong></td>
      <td>90.67</td>
      <td>88.29</td>
      <td><strong>92.14</strong></td>
      <td>85.47</td>
      <td><strong>89.90</strong></td>
    </tr>
    <tr>
      <td>Sparseness</td>
      <td><strong>42.74</strong></td>
      <td>34.82</td>
      <td>35.26</td>
      <td><strong>49.70</strong></td>
      <td>31.49</td>
      <td><strong>39.92</strong></td>
    </tr>
    <tr>
      <td>Relevance Mass Accuracy</td>
      <td><strong>50.28</strong></td>
      <td>45.89</td>
      <td>46.19</td>
      <td><strong>53.32</strong></td>
      <td>42.96</td>
      <td><strong>50.13</strong></td>
    </tr>
    <tr>
      <td>Relevance Rank Accuracy</td>
      <td><strong>62.22</strong></td>
      <td>59.50</td>
      <td>58.13</td>
      <td><strong>61.44</strong></td>
      <td>53.64</td>
      <td><strong>64.69</strong></td>
    </tr>
    <tr>
      <td>Top-K Intersection</td>
      <td><strong>75.07</strong></td>
      <td>69.09</td>
      <td>71.21</td>
      <td><strong>76.59</strong></td>
      <td>63.83</td>
      <td><strong>71.68</strong></td>
    </tr>
    <tr>
      <td>AUC</td>
      <td><strong>83.12</strong></td>
      <td>80.45</td>
      <td>76.49</td>
      <td><strong>81.28</strong></td>
      <td>70.13</td>
      <td><strong>83.03</strong></td>
    </tr>
  </tbody>
</table>

<table>
  <caption>Heatmap performance comparison on PASCAL VOC 2012 (higher values are better).</caption>
  <thead>
    <tr>
      <th rowspan="2">Metric</th>
      <th colspan="2">Supervised (ResNet-50)</th>
      <th colspan="2">SSL (SimCLR)</th>
      <th colspan="2">SSL (SwAV)</th>
    </tr>
    <tr>
      <th>Grad-CAM</th>
      <th>LaFAM</th>
      <th>RELAX</th>
      <th>LaFAM</th>
      <th>RELAX</th>
      <th>LaFAM</th>
    </tr>
  </thead>
    <tbody>
    <tr><td>Pointing Game</td><td style="text-align:right">90.83</td><td style="text-align:right"><strong>91.23</strong></td><td style="text-align:right">91.63</td><td style="text-align:right"><strong>94.68</strong></td><td style="text-align:right">82.73</td><td style="text-align:right"><strong>93.62</strong></td></tr>
    <tr><td>Sparseness</td><td style="text-align:right"><strong>44.39</strong></td><td style="text-align:right">36.20</td><td style="text-align:right">36.04</td><td style="text-align:right"><strong>51.00</strong></td><td style="text-align:right">32.36</td><td style="text-align:right"><strong>41.61</strong></td></tr>
    <tr><td>Relevance Mass Accuracy</td><td style="text-align:right"><strong>40.44</strong></td><td style="text-align:right">37.13</td><td style="text-align:right">38.00</td><td style="text-align:right"><strong>45.67</strong></td><td style="text-align:right">34.18</td><td style="text-align:right"><strong>42.17</strong></td></tr>
    <tr><td>Relevance Rank Accuracy</td><td style="text-align:right">53.53</td><td style="text-align:right"><strong>53.94</strong></td><td style="text-align:right">54.55</td><td style="text-align:right"><strong>58.73</strong></td><td style="text-align:right">46.40</td><td style="text-align:right"><strong>61.05</strong></td></tr>
    <tr><td>Top-K Intersection</td><td style="text-align:right"><strong>65.46</strong></td><td style="text-align:right">63.42</td><td style="text-align:right">67.82</td><td style="text-align:right"><strong>75.05</strong></td><td style="text-align:right">56.22</td><td style="text-align:right"><strong>67.63</strong></td></tr>
    <tr><td>AUC</td><td style="text-align:right">82.87</td><td style="text-align:right"><strong>84.00</strong></td><td style="text-align:right">79.88</td><td style="text-align:right"><strong>85.33</strong></td><td style="text-align:right">71.24</td><td style="text-align:right"><strong>87.74</strong></td></tr>
  </tbody>
</table>


<p>
<strong>Pointing Game:</strong> Percentage of samples where the single most salient pixel falls inside the ground-truth object region.
</p>

<p>
<strong>Top-$K$ Intersection:</strong> Fraction of the top $K\%$ most salient pixels that lie within the true object region.
</p>

<p>
<strong>Relevance Mass Accuracy / Rank Accuracy:</strong> Metrics that assess how much of the total saliency "mass" falls within the object mask, or how well the saliency values are ordered (foreground > background).
</p>

<p>
<strong>Sparseness:</strong> A measure of how concentrated the heatmap is. Higher means the map is tight and focused; lower means it is more diffuse or noisy.
</p>

<p>
<strong>AUC (Area Under the Curve):</strong> Treats every pixel’s saliency value as a prediction of “foreground vs. background” and measures how well these values separate the two groups. A high AUC means the object pixels generally have higher values than background pixels.
</p>

These measurements show that channel-averaged feature activity can localize annotated objects in the evaluated CNNs, including the self-supervised models. They do not establish that every active feature is an object detector or that the network has discarded all background information. Nor does a bright region under a wrong prediction prove that the model recognized the object correctly.

The metric choices also matter. Sparseness is not a measure of correctness. The localization metrics do use annotations and can penalize responses to real but unannotated objects. This becomes especially important when we compare methods that produce very different amounts of fine detail, as we will see [below](#a-note-on-evaluation).

</details>


## A Finer Picture

The coarse grid comes from the encoder itself. CNNs progressively downsample the image, and ViTs cut it into fixed-size patches. To obtain a finer map, we can either upsample the features with AnyUp or propagate the activations back to the input using LRP. Both approaches increase spatial detail, but they do so in very different ways.

AnyUp [<a href="#wimmer2025anyup" data-ref="wimmer2025anyup">Wimmer et al.</a>] is a pretrained feature upsampler designed to work with features from different vision encoders. It takes low-resolution features together with the corresponding higher-resolution image and upsamples the feature tensor to the image resolution. During training, the encoder's features for a small crop serve as a finer-scale target for the same region of the full image, so AnyUp learns to predict what the features would look like at a higher resolution. The image provides spatial guidance that ordinary interpolation does not have.

For example, ConvNeXt turns a $224\times224$ image into a $1024\times7\times7$ feature tensor. We can give AnyUp this tensor together with the image, and it produces a $1024\times224\times224$ feature tensor. We can average its channels in the same way as before to obtain a full-resolution heatmap. Under the hood, every output feature vector is a weighted average of the coarse feature vectors in a small window around it, with the weights computed by attention between the image pixels and the coarse features. When the features are already non-negative, as in a ResNet, the same holds for the heatmap, since averaging over channels is linear as well. Each pixel of the refined map is then a weighted average of nearby cells of the coarse map. AnyUp decides where the boundaries between them run, but it cannot create a response the coarse map does not have.

Layer-wise relevance propagation (LRP) [<a href="#bach2015on" data-ref="bach2015on">Bach et al.</a>, <a href="#montavon2018methods" data-ref="montavon2018methods">Montavon et al.</a>] takes a different approach. Instead of increasing the resolution of the feature tensor, it traces relevance backward through the encoder. At each layer, relevance is redistributed to the preceding neurons according to a chosen propagation rule until it reaches the input, producing a pixel-level attribution map. It was originally designed to explain predictions from supervised models by redistributing the relevance of a target output back to the input. But nothing prevents us from choosing a different starting signal. Instead of a class score, we can start from the encoder activations themselves. These activations already reflect the successive filtering performed by the network, where weaker or less relevant responses may have been suppressed along the way. Propagating them backward therefore reveals which input pixels contributed to the features that remained active at the chosen layer.

The figure below shows the CNNs, two transformers that restrict most of their attention to local windows, and three plain vision transformers. Three more plain ViTs follow in a separate figure further down, because the same reduction fails for them.

{% include figure-gallery.html id="lafam-encoders" data=site.data.galleries.lafam_encoders %}

The Averaging column is blocky by construction, since each block is one feature cell. An object shows up as a cluster of bright blocks that spills over its edges into the background. AnyUp keeps that pattern but redraws it to follow the image. The blocks around the two dogs become two dog silhouettes, and the cluster over the fish takes on the fish's outline. Each object comes out filled in almost evenly, with a sharp border wherever the image has one.

The glow that spills into the background can come from three places. At $7\times7$, a cell on the edge of an object covers part of the object and part of the background. Each cell also sees more of the image than its own square. In a CNN, each feature is computed from a wider region around it, and in a transformer, attention lets each token gather information from other tokens. A patch of grass next to a dog can therefore carry information about the dog. Finally, the background can matter in its own right. Grass, trees, and sky are things an SSL model can learn to recognize, just like dogs. The background may also serve as a shortcut, as when a model learns to expect a dog wherever it sees a lawn. AnyUp cannot tell these cases apart. Each refined value is a weighted average of nearby coarse values, so AnyUp keeps whatever the coarse map contains and only gives it sharper edges.

LRP shows a different picture. Its relevance concentrates on image structures such as contours, object parts, and internal edges, and little of it reaches the background. The maps are sparse, and they are not free of artifacts either. The CNN attribution maps are generally spatially continuous, whereas Swin shows visible block structure in some of the images. The plain ViTs show blocks the size of their patches, and for MAE a regular grid dominates much of the relevance map. Producing a pixel-level attribution map therefore does not remove the spatial constraints introduced by patch embeddings or by the chosen propagation rules.

One caveat is that the [Zennit](https://github.com/chr5tphr/zennit) library I used provides built-in support mainly for VGG and ResNet architectures. ConvNeXt and ViTs contain operations for which I had to implement additional propagation rules, as others have done for transformers [<a href="#achtibat2024attnlrp" data-ref="achtibat2024attnlrp">Achtibat et al.</a>]. These LRP attribution maps should therefore be read as the result of one reasonable set of propagation choices. I will go through these choices, and how much they change the maps, in an upcoming post.

<details>
<summary id="a-note-on-evaluation">A note on evaluation</summary>
<p>
The original LaFAM paper evaluated attribution largely as an object-localization problem, using segmentation masks as ground truth. This works reasonably well for comparing coarse maps, but becomes harder to interpret once the maps contain fine spatial structure. Mask-based localization treats the annotated object region as the target, without distinguishing which parts of that object actually contributed to the representation. A fine-grained map of a bicycle, for example, might concentrate on its frame, wheels, or other structures, while a smoother map spreads its response across much of the object. Some localization metrics can favor the latter because more of the annotated region receives high attribution, even though the map contains less spatial detail. Thin structures may also be imperfectly represented in the segmentation mask itself. Label-free maps introduce another mismatch: responses to other objects are counted as background even when the encoder genuinely responds to them.
</p>
</details>

### A Watermark in the Web

The spider web image holds a small surprise. In its bottom-right corner sits a tiny sparkle, the watermark of the image generator that created the picture. It covers less than one percent of the image and is easy to miss, but several encoders do not miss it. This is the kind of finding that makes label-free maps useful for debugging. Nobody asked the models about the watermark, and no label points to it, yet the maps show that the models respond to it.

The two methods reveal it in different ways. For CLIP ConvNeXt, the watermark's cell is the brightest of all 49 cells in the averaged map, and LRP places about five percent of the relevance on it. MaxViT barely reacts to the watermark in the averaged map, but its LRP map puts almost eight percent of the relevance there, with a peak about four times higher than anywhere else. Since every map is scaled by its own maximum, that single spot turns the rest of the MaxViT map nearly black.

{% include heatmap-grid.html id="lafam-watermark" data=site.data.galleries.lafam_watermark %}

Once we know about the watermark, we can set it aside and look at the rest. For CLIP ConvNeXt, it is enough to leave the watermark's cell out of LRP's starting amount. The share on the watermark then drops from five to about two percent, and the center of the web becomes visible, although the map stays noisy. For MaxViT, the same step changes almost nothing, because relevance reaches the watermark from cells across the whole image, even from the opposite corner. Only when the map is scaled without the watermark does it become clear that the threads of the web were there all along.

The example also shows where the maps stop. They tell us where the representation responds, but not what the response is about. We recognized the watermark because we looked at the image ourselves. To the maps, it is just another bright region, like the web. Telling such responses apart without labels requires asking the representation something else, for instance which parts of an image it treats as alike. Why a small watermark draws so much of the response is a question of its own, which we leave for a later post.

## When ReLU Is Not Enough

The first figure includes three plain vision transformers, DINOv1, DINOv3 and MAE, and their maps behave much like those of the CNNs. For three other plain ViTs, a supervised ViT-B/16, a CLIP ViT-B/16 and DINOv2, the same reduction fails. Their maps are bright almost everywhere apart from a handful of much darker tokens, and the objects are hard to find.

{% include figure-gallery.html id="lafam-reductions" data=site.data.galleries.lafam_reductions %}

What sets the two groups apart? The obvious suspect is the ReLU, but it does not throw most of the values away. In every plain ViT we looked at, about half of all feature values are positive, just as in the CNNs. Two other differences stand out.

The first is the dark tokens. All three encoders that fail have some of them in the background of every image, while DINOv1, DINOv3 and MAE have none. Such outlier tokens are a known effect in ViTs [<a href="#darcet2024vision" data-ref="darcet2024vision">Darcet et al.</a>, <a href="#sun2024massive" data-ref="sun2024massive">Sun et al.</a>]. Darcet et al. found them in DINOv2, a supervised ViT and OpenCLIP, but not in the original DINO. They also showed that a few extra tokens, called registers, remove them, and DINOv3 has four such registers. A heatmap is stretched between its darkest and its brightest token, so these few dark tokens use up most of the color scale, and nearly all other tokens end up bright.

The second is the channels that never switch off. In a CNN, a channel is usually active in some parts of the image and inactive in others. In a ResNet, inactive means zero, and in ConvNeXt it means negative. In the supervised ViT and CLIP, between one channel in eight and one in six stays positive across almost the whole image, and these channels are strongest on the background. The ReLU passes them unchanged, so they pull the map away from the objects. The other plain ViTs have such channels too, but too few or too weak to shape the map. For the supervised ViT, leaving out these channels and setting the color scale without the outlier tokens is enough to bring the objects back.

The remedy is to center each channel before the ReLU. From every channel, we subtract its average over all tokens of the image. The channel is then positive exactly where it is higher than usual in this image, whatever its overall level. Like the plain average, this needs no labels, no other images and no backpropagation.

For the three encoders that failed, centering helps, but the maps stay far from clean. Only the image of the dog and the cat and the image with six objects come out readable. In the image of the dog and the cat, both animals stand out for CLIP and DINOv2, while the supervised ViT shows the cat clearly and the dog only faintly. The other images stay noisy. The remaining encoders work without centering, which is why the first figure keeps the original reduction for all of them.

In the photo of the fish, the supervised ViT shows the fish darker than the person holding it. The classifiers may not need the fish much. If we remove it from the photo, every classifier in this post still predicts a tench. ImageNet photos of tench usually show an angler holding the catch, so the models have learned the scene rather than the fish. This is a known shortcut [<a href="#geirhos2020shortcut" data-ref="geirhos2020shortcut">Geirhos et al.</a>], and finding such shortcuts is exactly what attribution methods were designed for.

<details>
<summary id="a-closer-look-at-the-channels-and-the-outlier-tokens">A closer look at the channels and the outlier tokens</summary>
<p>
A channel counts as never switching off here if it is positive at nine out of ten tokens of an image or more. All six plain ViTs have such channels, but only in the supervised ViT and CLIP are they numerous enough to shape the map.
</p>
<p>
Why ViTs have such channels, we can only guess. A ResNet applies a ReLU itself, so zero is a meaningful dividing line for its features. A ViT does not, and for many of its channels zero seems to be an arbitrary point. ConvNeXt has no ReLU at its output either, but its channels cross zero within an image, which fits with its maps working without centering.
</p>
<p>
Centering only shifts the channels and does not rescale them. Dividing each channel by its spread as well gave almost the same results in our tests, so the shift is what matters.
</p>
<p>
The outlier tokens are a separate problem. Setting the color scale without them brings back the contrast, but not clear objects. They are also not weak inside the network. Just before the output, each of them has one channel with an enormous value, the same channel in every image. In the supervised ViT, this value reaches about a thousand, while the largest value of an ordinary token is a few dozen.
</p>
<p>
The last step of these encoders is a layer normalization. It divides each token by the spread of its values, so that all tokens end up on a similar scale. In an outlier token, that spread comes almost entirely from the one huge channel. Dividing by it leaves the spike as the only large value and pushes all other channels close to zero. The token becomes nearly flat. The ReLU map counts only values above zero, and a flat token has very little of them, so it comes out darkest.
</p>
<p>
Centering compares each channel with its own average in the image instead of with zero. In the supervised ViT, about a third of the channels are clearly negative on average over the image. For them, a value near zero is well above average. A flat token therefore scores high in all these channels at once, and some of the outlier tokens become bright spots in the centered maps. DINOv2 has fewer such channels, and its outlier tokens stay dark. In neither map does their brightness tell us anything about the image at their position. Only the reference they are compared with has changed.
</p>
</details>

AnyUp works well on the centered features. It turns the token grid into object silhouettes, much as it does for the CNNs, and its maps show the objects at least as clearly as the centered maps they come from. LRP does not use this reduction at all, so it needs no remedy. For these three encoders, its maps show the objects, with the same patch-sized blocks as the other plain ViTs.

## Limitations

**Resolution.** The heatmap produced by averaging can only be as sharp as the feature grid being averaged, namely $7\times7$ for the CNNs here and $14\times14$ for ViT-B/16. On a $224\times224$ input, one cell of a $7\times7$ grid spans $32\times32$ pixels. Small objects may disappear into a single cell, and nearby objects may merge into one response. Both refinements add detail, but each brings its own assumptions.

- *AnyUp* takes its detail from the image. Its authors model each refined feature as a weighted average of nearby coarse features, a simplification they acknowledge themselves. The image decides the weights and aligns coarse responses with sharp image boundaries, but it cannot tell whether those boundaries reflect evidence the encoder used, so a background response can look just as precise as an object.

- *LRP* takes its detail from the network's own computation, but the result depends on where relevance starts and which propagation rules carry it. LRP deliberately allows different rules, and its authors leave the choice to the particular model, problem, or practical requirements instead of prescribing a general procedure. These choices can visibly change the maps.

**What does the heatmap mean?** Averaging collapses the feature dimension into a single value at each spatial location. This gives us a useful spatial summary, but it throws away information about which features produced the response. Two bright regions may be driven by entirely different features, and the map cannot tell an object from a background texture if both produce strong activations. This is particularly important for self-supervised models, where the individual feature dimensions have no predefined semantic meaning.

The reduction itself is also a choice. Here, we apply ReLU before averaging, so negative values are discarded, and as we saw above, this breaks down for some encoders unless each channel is centered first. We could instead average the signed values directly, average their absolute magnitudes, or summarize each feature vector by its norm. These operations emphasize different properties of the representation and need not produce the same heatmap. Why such simple summaries preserve coherent spatial structure at all is therefore not obvious.

Finally, a strong response does not tell us whether that response matters for a particular task. The averaging procedure has no prediction target, so a bright region may contain useful object information, contextual information, or a feature that a downstream classifier never uses. The map tells us where the representation responds, but not what the response represents or what it will ultimately be used for.

## Conclusion

Averaging across feature channels is almost embarrassingly simple. It ignores what the individual features mean, gives them all the same weight, and does not ask the model for a prediction. Yet the resulting maps repeatedly line up with coherent objects and other structures in the image.

Look back at the scenes with several objects, though. Supervised models trained with one label per image must name a single class. Yet their maps highlight the dog and the cat, both dogs, and all six objects. So what happens to everything they see but never name? 

The answer points to something deeper than our starting question, namely that models see far more than we ever asked them to. In my recent work [<a href="#karjauv2026rethinking" data-ref="karjauv2026rethinking">Karjauv</a>], I demonstrate that this is no accident. Classifiers with global average pooling and a linear head decompose cleanly into spatial evidence. Beneath the final prediction, the network quietly behaves like a multi-instance learner.

Then there is the background. If every location carries evidence, what makes the models respond strongly to objects, while "irrelevant" regions stay quiet?

SSL models never even get that one label. So when their maps light up, what exactly are they responding to, and how would we know?

Those are the questions I will turn to next.

## References

{%- capture references -%}
Karjauv, A., et al. | 2024 | LaFAM: Unsupervised Feature Attribution with Label-free Activation Maps;
Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K.-R., & Samek, W. | 2015 | On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation;
Montavon, G., Samek, W., & Müller, K.-R. | 2018 | Methods for interpreting and understanding deep neural networks;
Meehan, C., et al. | 2023 | Do ssl models have déjà vu? a case of unintended memorization in self-supervised learning;
Achtibat, R., et al. | 2024 | AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for Transformers;
Wimmer, T., et al. | 2025 | AnyUp: Universal Feature Upsampling;
Darcet, T., Oquab, M., Mairal, J., & Bojanowski, P. | 2024 | Vision Transformers Need Registers;
Sun, M., Chen, X., Kolter, J. Z., & Liu, Z. | 2024 | Massive Activations in Large Language Models;
Raghu, M., Unterthiner, T., Kornblith, S., Zhang, C., & Dosovitskiy, A. | 2021 | Do Vision Transformers See Like Convolutional Neural Networks?;
Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. | 2020 | Shortcut Learning in Deep Neural Networks;
Karjauv, A. | 2026 | Rethinking Global Average Pooling: Your Classifier Is Secretly a Multi-Instance Learner;
{%- endcapture -%}

{% include reference.html ref=references %}
