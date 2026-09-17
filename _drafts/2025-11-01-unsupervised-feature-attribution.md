---
layout: post
styles:
  - /assets/css/lafam.css
title: Unsupervised Feature Attribution for Foundation Models
date: 2025-11-01 00:00:00 +0200
tags: XAI, CNN, Foundation Models, Self-Supervised Learning
description: How activation maps and relevance propagation can shed light on deep models without any labels.
---

Foundation models trained without labels can learn a lot, but understanding what they actually see is another story. Convolutional networks and transformers, though, hide a clue: their activations quietly reveal where the model is looking. No labels, no gradients, just the network explaining itself. The only problem is that its vision is still a bit blurry… for now.

<!--more-->

This journey started with my attempt to use Generative Adversarial Networks (GANs) as feature extractors for medical images. GANs can synthesize high-quality images from random noise and control the generation process via latent variables, but they are not inherently invertible. I tried to address this limitation, but it didn’t work well (see my earlier [post](/2023/01/06/turning-stylegan-into-a-latent-feature-extractor.html) for details). Debugging that encoder-based GAN quickly turned into an exercise in frustration, exposing how few tools exist to reveal what such models actually learn. That limitation led to a broader question -- how can we explain models that where trained without explicit annotations? This question is central to self-supervised learning (SSL), where models learn latent representations directly from unlabeled data. How can we see what such models focus on? That question brings us to LaFAM [<a href="#karjauv2024lafam" data-ref="karjauv2024lafam">Karjauv et al.</a>], a straightforward but effective method for unsupervised feature attribution.

This post explores whether the model has already computed a useful spatial explanation while processing the image. We will look how LaFAM provides quick saliency maps in an unsupervised setting, born out of the need to interpret SSL models. This approach allows us to visualize where learned features respond without choosing a class and computing gradients. We also compare LRP with AnyUp for refining the resolution of spatial explanations.
<!-- Along the way, we'll maintain a critical lens: Are our evaluation metrics actually fair? What limitations do these methods have? -->

<!-- By the end, we’ll see not only where the network looks, but also reflect on the perennial XAI question of “so what?” (or rather what is it seeing there?). -->

<!-- (If you're just joining, no worries – each part stands alone. But know that Part 1 dealt with GANs and encoder limitations, and Part 3 will tackle Multiple Instance Learning. Now let's dive in!) -->

## Self-Supervised Learning Meets XAI

Self-Supervised Learning (SSL) has emerged as a way for models to learn useful representations without manual labels. Vision models like SimCLR and DINO can train on millions of images by solving proxy tasks (e.g., contrasting different augmented views of the same image) and then be fine-tuned for actual tasks. SSL models are often called foundation models for their broad adaptability. Yet, the absence of labels makes it difficult to verify whether the learned features are actually relevant for a given downstream task. Moreover, when trained with proxy tasks such as random cropping, a model may unintentionally associate irrelevant features without us realizing, and evaluating it is non-trivial [<a href="#meehan2023do" data-ref="meehan2023do">Meehan et al.</a>].

Explainable AI (XAI) offers ways to probe a model’s reasoning, for example, by producing saliency maps that highlight important regions of an input. Traditional XAI methods, though, are designed for supervised models by attributing input importance for a specific class. Methods like Grad-CAM and occlusion-based methods (e.g., RISE) require a so-called score function. This function takes a target class as input to guide the attribution. This poses a problem for SSL, as there are no explicit labels to explain.

There have been attempts to adapt XAI to label-free models. One such attempt is RELAX (Representation Learning Explainability) which extended the supervised occlusion method RISE. The key idea is clever: SSL models output embeddings that are unitless, meaning that each value does not refer to any particular feature. Since we don't know what a target embedding should look like, the authors propose to first create a reference embedding from the original input and extract embeddings from occluded inputs. We can then define a score function that measures cosine similarity between the reference embedding and the occluded ones and use it to attribute the most salient features.

However, RELAX ended up being computationally expensive as it needs many forward passes with different masks, and it often produced very noisy maps. Moreover, using random patch occlusions can introduce unnatural artifacts, leading the model to react strangely to a big gray patch that it would never see during normal operation.

## Label-Free Activation Maps

<!-- An important advantage of CNNs is that their spatial feature maps preserve the structure of the input image. As layers stack, each convolution processes a local region of the previous feature map, which causes receptive fields to expand with depth. By the final convolutional layer, neurons cover enough of the image to encode class-specific signals while still retaining coarse spatial layout. -->

A key strength of CNNs is that their activation maps maintain a connection between detected patterns and their positions in the input image. This property makes the models inherently more explainable and underpins the success of Class Activation Map (CAM) methods. By weighting the maps in the final convolutional layer according to their contribution to a target class, these methods can localize the image regions most relevant in supervised settings.

<details><summary>What is an activation map?</summary>
<p>
Each convolutional layer takes an input and produces activation maps (also called feature maps) that record activations of specific visual patterns across the image. The first layer processes raw pixels and responds to simple local features such as edges or color contrasts. Each subsequent layer takes the feature map from the previous one and combines these basic patterns into more complex and abstract representations, like textures or object parts. The area a neuron responds to is called its receptive field. As the network goes deeper, receptive fields grow as a result of pooling operations or convolution strides, which reduce the spatial size of feature maps. This lets deeper neurons capture a larger portion of the image while still preserving coarse spatial relationships.
</p>
</details>


But what if we don’t have a class? The answer is simple -- we don't need it. We can simply average all the activation maps at the last convolutional layer to get a generic saliency map. This label-free map doesn’t focus on any one class. It treats every learned feature as equally interesting, highlighting regions that strongly activate any of the high-level features in that layer. Essentially, it’s a visualization of “where the network is looking” in a class-agnostic sense.

LaFAM (Label-free Activation Map) [<a href="#karjauv2024lafam" data-ref="karjauv2024lafam">Karjauv et al.</a>] evaluates this approach systematically. The method is astonishingly simple yet effective. It outperforms RELAX and even stands up well against Grad-CAM.

Think of the feature channels as a collection of sensors. Each responds to a different learned pattern. LaFAM averages their readings at each location, showing where the collection responds strongly. We are reading a signal the network already produced. When we train a model, it sees similar objects and learns to associate them with certain features that are almost always present, while unrelated background features usually do not correlate with the object and the model learns to ignore them. The resulting saliency map reflects the model’s learned feature preferences.

### A Note on Vision Transformers (ViTs)

ViTs do not have the same spatial structure as CNNs. However, an image in a ViT is split into patches called tokens. Each token naturally corresponds to a specific region of the image. The attention mechanism in ViTs allows each token to interact with others, enabling them to exchange information and capture global context. Tokens from the final layer can be averaged to produce a saliency map similar to CAM and, hence, LaFAM can be applied to ViTs as well. However, most of the work on ViTs has focused on attention-based methods and require a target class to produce saliency maps. To best of my knowledge, there is no systematic evaluation of CAM-based methods on ViTs yet. 

### Qualitative Comparison

<figure>
<img src="/assets/img/posts/lafam/pascal_voc_2012_results.svg" alt="LaFAM vs. Grad-CAM vs. RELAX" />
  <figcaption>Comparison on PASCAL VOC 2012. LaFAM produces maps similar to Grad-CAM but remains robust when the model misclassifies (first row).</figcaption>
</figure>

In the figure above we see this comparison in action. This figure compares three attribution methods: LaFAM, RELAX, and Grad-CAM as a baseline. LaFAM produces very similar saliency maps to those produced by Grad-CAM in supervised models (i.e., applied to the same supervised model).

In the first row, the model actually mispredicted the ImageNet class, so Grad-CAM dutifully highlighted an irrelevant region associated with the wrong class. This illustrates a subtle strength--by not being yoked to the top-1 predicted label, simply averaging the last CNN layer won’t completely go off the rails when the model’s prediction is off. It shows everything the model found salient, not just what influenced the (possibly incorrect) class choice. For SSL models, LaFAM is notably less noisy than RELAX’s outputs.

The next figure demonstrates that LaFAM, being class-agnostic, highlights multiple objects in the image (e.g., both dogs), whereas Grad-CAM focuses on just one object tied to the predicted class. This multi-object sensitivity is a useful trait in many scenarios, as real-world images almost always contain several relevant items.

![LaFAM for Multiple Objects](/assets/img/posts/lafam/pascal_voc_2012_2_classes.svg){: .center-image }

Let's look closer at the model misprediction cases. For Grad-CAM we visualize the saliency map for the (wrong) predicted class. As a result, Grad-CAM highlights irrelevant regions, while LaFAM attributes all learned features. This again underscores the advantage of being label-free: LaFAM reflects what the model finds salient overall, not just what it thinks is the “correct” class.

![Grad-CAM for Misclassification Case](/assets/img/posts/lafam/imgnet_missclf_short.svg){: .center-image }

LaFAM clearly highlights true objects, suggesting that these objects strongly activate multiple channels in the final conv layer. However, the prediction is wrong. The reason could be that the last fully connected layer puts more weight on specific features that mislead the final decision, even though many other features correctly identify the object.

<details markdown="1">
<summary>Supporting evidence: the original LaFAM evaluation</summary>

The LaFAM paper [<a href="#karjauv2024lafam" data-ref="karjauv2024lafam">Karjauv et al.</a>] systematically evaluates and compares averaged activation maps against RELAX (the prior SSL method) using SSL models (SimCLR and SwAV on ResNet50 backbones), and also compared it against Grad-CAM for a fully supervised ResNet50 classifier as a sanity check. The saliancy maps for ResNet50 have only 7x7 size and were upscaled to match the input image size using nearest-neighbor interpolation. Since we don't have class labels to evaluate “correctness” of an explanation, the evaluation was performed on datasets with segmentation masks (ImageNet-S and PASCAL VOC) and a suite of evaluation metrics from the Quantus XAI evaluation framework. Essentially, we treat it as a localization task, whith goal to assess how well the saliency maps align with the segmentaions masks.

<table id="imagenet_results">
  <caption>Saliency maps performance comparison on ImageNet-S (higher values are better).</caption>
  <thead>
    <tr>
      <th rowspan="2">Metric</th>
      <th colspan="2">Supervised (ResNet50)</th>
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
      <td>Pointing-Game</td>
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
  <caption>Saliency maps performance comparison on PASCAL VOC 2012 (higher is better).</caption>
  <thead>
    <tr>
      <th rowspan="2">Metric</th>
      <th colspan="2">Supervised (ResNet50)</th>
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
    <tr><td>Pointing-Game</td><td style="text-align:right">90.83</td><td style="text-align:right"><strong>91.23</strong></td><td style="text-align:right">91.63</td><td style="text-align:right"><strong>94.68</strong></td><td style="text-align:right">82.73</td><td style="text-align:right"><strong>93.62</strong></td></tr>
    <tr><td>Sparseness</td><td style="text-align:right"><strong>44.39</strong></td><td style="text-align:right">36.20</td><td style="text-align:right">36.04</td><td style="text-align:right"><strong>51.00</strong></td><td style="text-align:right">32.36</td><td style="text-align:right"><strong>41.61</strong></td></tr>
    <tr><td>Relevance Mass Accuracy</td><td style="text-align:right"><strong>40.44</strong></td><td style="text-align:right">37.13</td><td style="text-align:right">38.00</td><td style="text-align:right"><strong>45.67</strong></td><td style="text-align:right">34.18</td><td style="text-align:right"><strong>42.17</strong></td></tr>
    <tr><td>Relevance Rank Accuracy</td><td style="text-align:right">53.53</td><td style="text-align:right"><strong>53.94</strong></td><td style="text-align:right">54.55</td><td style="text-align:right"><strong>58.73</strong></td><td style="text-align:right">46.40</td><td style="text-align:right"><strong>61.05</strong></td></tr>
    <tr><td>Top-K Intersection</td><td style="text-align:right"><strong>65.46</strong></td><td style="text-align:right">63.42</td><td style="text-align:right">67.82</td><td style="text-align:right"><strong>75.05</strong></td><td style="text-align:right">56.22</td><td style="text-align:right"><strong>67.63</strong></td></tr>
    <tr><td>AUC</td><td style="text-align:right">82.87</td><td style="text-align:right"><strong>84.00</strong></td><td style="text-align:right">79.88</td><td style="text-align:right"><strong>85.33</strong></td><td style="text-align:right">71.24</td><td style="text-align:right"><strong>87.74</strong></td></tr>
  </tbody>
</table>


<details><summary>What are these metrics?</summary>

<p>
<strong>Pointing Game:</strong> Percentage of samples where the single most salient pixel falls inside the ground-truth object region.
</p>

<p>
<strong>Top-$K$ Intersection:</strong> Fraction of the top $K%$ most salient pixels that lie within the true object region.
</p>

<p>
<strong>Relevance Mass Accuracy / Rank Accuracy:</strong> Metrics that assess how much of the total saliency "mass" falls within the object mask, or how well the saliency values are ordered (foreground > background).
</p>

<p>
<strong>Sparseness:</strong> A measure of how concentrated the saliency map is. Higher means the map is tight and focused; lower means it is more diffuse or noisy.
</p>

<p>
<strong>AUC (Area Under the Curve)</strong>: Treats every pixel’s saliency value as a prediction of "foreground vs background" and measures how well they separate. A high AUC means the object pixels generally have higher values than background pixels.
</p>
</details>

### Reading the original results

These measurements show that channel-averaged feature activity can localize annotated objects in the evaluated CNNs, including the self-supervised models. They do not establish that every active feature is an object detector or that the network has discarded all background information.

The class-free view is useful when a predicted label is wrong: it lets us inspect activity without first selecting that mistaken class. A region lighting up is still not proof that the model recognizes its object correctly. Reading the class evidence at individual locations is the subject of the next post.

The metric choices also matter. Sparseness measures how concentrated a map is, but it is not a measure of correctness. The localization metrics do use annotations, and can penalize responses to real but unannotated objects. This becomes especially important when we compare methods that produce very different amounts of fine detail.


</details>

LaFAM's limitation is easy to see: the final ResNet-50 features occupy a $7\times7$ grid. The response is already informative, but its display is coarse. Nearest-neighbor resizing makes larger tiles, and bilinear resizing blends their edges. To refine the spatial detail, we can use the image itself as guidance.

## A finer picture with AnyUp

[AnyUp](https://arxiv.org/abs/2510.12764) is a pretrained feature upsampler that can work with features from different encoders. It receives both the low-resolution features and a higher-resolution image, then produces features on a finer grid. The image supplies visual guidance that ordinary interpolation lacks. [Official implementation](https://github.com/wimmerth/anyup).

The analogy is redrawing a coarse sketch while looking at a sharp photograph. The photograph can help place boundaries, even though the original sketch did not resolve them.

Here, I give AnyUp the complete $2048\times7\times7$ feature tensor and the same $224\times224$ image used by the encoder. Only after upsampling do I average the channels to form LaFAM. Averaging before AnyUp would change its feature-dependent guidance, so it would be a different experiment.

The spatial signal still comes from the encoder's features. AnyUp adds a pretrained model that guides their enlargement; using it requires no backward attribution pass or retraining of the encoder. The resulting detail is image-guided refinement of the coarse representation.

### Compare maps across encoders

Pick one of five real images from the strip below, then compare both LaFAM and its AnyUp refinement across all ten encoders. Adjust the map opacity, and move over either view to reveal the photograph in a small region around the pointer.

{% include figure-gallery.html id="lafam-encoders" data=site.data.galleries.lafam_encoders %}




In the opening image, the fish already produces a strong response in the coarse LaFAM map. AnyUp gives that response a finer spatial shape. The object-related activity was present before the refinement; we did not need LRP to reveal it.

The examples were chosen by their positions in a fixed sample, before comparing scores. The same images appear for every backbone and method. They are illustrative cases, not a collection of each method's best results.

### Do we need to propagate relevance backward?

LRP offers a different view: it redistributes relevance through the encoder to input pixels according to chosen rules. That can be useful when the question calls for such an attribution. It is also extra machinery: a starting relevance assignment, rules for the network's operations, and a backward pass.

The LaFAM examples show that **a useful spatial explanation is already available before adding that machinery**. AnyUp offers a way to refine its resolution while keeping the explanation based on the forward features. This is the practical point of the comparison; it does not require LaFAM to win every attribution metric.

<details class="lf-detail" markdown="1">
<summary>How the label-free LRP comparison was defined</summary>

### LRP: follow the computation back

Ordinary LRP starts with relevance assigned to a class output, then redistributes it backward through the network. The redistribution rules determine how relevance is divided among a layer's inputs. For a label-free version, we can start at the encoder's features instead of a class score. No class name is needed. [Zennit documentation](https://zennit.readthedocs.io/en/latest/getting-started.html).

Think of following a finished meal back through its recipe. Which ingredients participated in which intermediate steps? LRP makes a rule-based accounting of the network's computation. It is more specific than making the final feature map larger, but it also requires decisions about how to handle each operation.

My exploratory notebooks used a mean-feature target. For each image, average its final feature values and apply the modified backward pass. With Zennit hooks installed, the backward operation follows the selected relevance rules rather than ordinary gradient rules.

There is an easy detail to miss here. Differentiating that mean gives each final feature an **equal starting amount of relevance**. Starting with relevance proportional to the feature's activation is a different choice. The second gives stronger features a larger initial share. Calling both simply “label-free LRP” hides a consequential difference.

For the comparison below I therefore keep two named variants: **uniform seed**, following the mean-output notebook experiment, and **activation seed**, using the same rules with a different initialization. These are new experiments extending the blog; the original LaFAM paper did not report an LRP benchmark.

<details class="lf-detail" markdown="1">
<summary>The exact LRP choices</summary>

Both variants use Zennit's `EpsilonPlusFlat` rules and `ResNetCanonizer`, which handles the ResNet's batch normalization and residual additions. The latter is an explicit correction to the notebook's incomplete network handling. Neither variant uses the classification head.

For the uniform seed, the essential calculation is:

```python
with composite.context(encoder):
    features = encoder(images)
    per_image_score = features.flatten(1).mean(1)
    relevance, = torch.autograd.grad(per_image_score.sum(), images)
```

Each image gets its own mean. Summing those individual scores allows a single backward call; it does not average the images together. The actual evaluation processes one image at a time.

For the activation seed, the backward vector is `features.detach() / features[0].numel()`. Relevance starts proportional to each activation rather than uniformly. For both variants, input-channel relevances are averaged into a pixel map, negative values are clipped, and the result is min-max scaled to $[0,1]$. There is no added smoothing.

</details>


</details>

### What the evaluation tells us

The new experiment uses 1,000 ImageNet validation images and their original bounding boxes. Ordinary LaFAM is a strong localization baseline in this comparison, and AnyUp produces finer maps without a backward attribution pass. The complete results include LRP, resizing baselines, and measured computation times.

Box scores measure coverage of rectangles. They do not directly measure the quality of fine boundaries: a smooth response can score well by filling empty space inside a box. Our sensitivity checks confirm that smoothing can raise the score without adding evidence. We therefore use the numbers alongside the real maps, keeping the distinction between spatial activity, visual refinement, and attribution faithfulness explicit.

<details class="lf-detail" markdown="1">
<summary>Full results and metric checks: boxes, smoothing, and LRP choices</summary>

### The ImageNet box comparison

For this extension, I reran the methods on **one preselected validation image from each of ImageNet's 1,000 classes**. The supervised ResNet-50, SimCLR, and SwAV encoders use the checkpoint families from the original LaFAM evaluation. The comparison adds AnyUp's original paper checkpoint; it does not train or tune an upsampler on these images.

Every method sees the same square center crop at $224\times224$. The original image's boxes receive the identical crop and resizing. I fill the union of the boxes to obtain a binary region for evaluation. No Pascal images or ImageNet-S masks enter this experiment.

This is a **sampled bounding-box comparison**, separate from the segmentation-mask tables earlier in the post. Its numbers should not be placed beside those tables as if the annotation, sample, and metric conventions were unchanged.

All 1000 selected images retained both inside and outside pixels after cropping. No images were excluded, and all three metrics use the same 1,000 images for every method and encoder.

The main score is **box AUROC**: pick an inside pixel and an outside pixel; how often does the inside pixel receive the higher map value? Ties count as half. Calculate the score separately for each image, then average across images. A score of 50% corresponds to chance ordering, while 100% means all inside pixels outrank all outside pixels.

<div class="lf-table-scroll" markdown="1">

| Encoder | LaFAM · nearest | LaFAM · bilinear | LRP · uniform seed | LRP · activation seed | LaFAM · AnyUp |
| --- | ---: | ---: | ---: | ---: | ---: |
| Supervised | 79.25% | 80.66% | 67.27% | 74.79% | 80.23% |
| SimCLR | 80.83% | 83.20% | 65.67% | 78.37% | 82.41% |
| SwAV | 79.74% | 82.18% | 63.63% | 73.77% | 80.99% |

</div>

Three patterns stand out. **AnyUp scores higher than either LRP variant on box AUROC for all three encoders.** Its advantage over activation-seeded LRP is 5.44 percentage points for the supervised encoder, 4.03 for SimCLR, and 7.21 for SwAV. The paired 95% image-bootstrap intervals are [4.36, 6.46], [3.22, 4.84], and [6.30, 8.17] points, respectively.

**Bilinear LaFAM has the highest box AUROC in this experiment.** AnyUp improves on nearest-neighbor resizing, but falls below bilinear resizing by 0.43, 0.80, and 1.20 points. The corresponding paired intervals are [−0.52, −0.34], [−0.90, −0.69], and [−1.32, −1.07] points for AnyUp minus bilinear. Learned upsampling is not automatically a better solution to this particular localization task.

**The LRP seed matters.** Activation-seeded LRP improves mean box AUROC over the uniform seed by roughly 8–13 points. That is a much larger change than the gap between AnyUp and bilinear resizing. A comparison that reports only “LRP” would leave out one of its most influential choices.

Other scores add nuance. For the supervised encoder, activation-seeded LRP places the largest fraction of its normalized relevance inside the boxes, even though its box AUROC and pointing score are lower. These scores ask different questions: ranking pixels, locating a maximum, and concentrating mass are not interchangeable.


<details class="lf-detail" markdown="1">
<summary>Pointing, relevance mass, timing, and uncertainty</summary>

**Pointing** checks whether the maximum lies inside the boxes. If several pixels share the maximum, I use the fraction of those pixels inside. This avoids choosing an arbitrary pixel within a nearest-neighbor tile. A completely flat map scores its annotated-area fraction rather than receiving an automatic hit.

**Relevance mass** is the fraction of the positive, normalized map inside the boxes. This depends on the map's normalization and on box size. All five methods use the same final positive clipping and min-max rule.

**Supervised ResNet-50**

<div class="lf-table-scroll" markdown="1">

| Method | Box AUROC | Pointing | Mass inside | Median time |
| --- | ---: | ---: | ---: | ---: |
| LaFAM · nearest | 79.25% | 84.46% | 66.05% | 2.9 ms |
| LaFAM · bilinear | 80.66% | 85.40% | 65.66% | 3.0 ms |
| LRP · uniform seed | 67.27% | 37.40% | 61.37% | 70.3 ms |
| LRP · activation seed | 74.79% | 59.90% | 68.81% | 69.5 ms |
| LaFAM · AnyUp | 80.23% | 84.80% | 66.21% | 40.2 ms |

</div>

**SimCLR ResNet-50**

<div class="lf-table-scroll" markdown="1">

| Method | Box AUROC | Pointing | Mass inside | Median time |
| --- | ---: | ---: | ---: | ---: |
| LaFAM · nearest | 80.83% | 87.38% | 71.44% | 3.0 ms |
| LaFAM · bilinear | 83.20% | 88.24% | 70.89% | 3.2 ms |
| LRP · uniform seed | 65.67% | 49.70% | 61.45% | 73.1 ms |
| LRP · activation seed | 78.37% | 80.30% | 69.74% | 71.9 ms |
| LaFAM · AnyUp | 82.41% | 88.61% | 71.58% | 41.0 ms |

</div>

**SwAV ResNet-50**

<div class="lf-table-scroll" markdown="1">

| Method | Box AUROC | Pointing | Mass inside | Median time |
| --- | ---: | ---: | ---: | ---: |
| LaFAM · nearest | 79.74% | 83.92% | 68.36% | 3.0 ms |
| LaFAM · bilinear | 82.18% | 84.51% | 67.76% | 3.1 ms |
| LRP · uniform seed | 63.63% | 51.80% | 60.85% | 72.4 ms |
| LRP · activation seed | 73.77% | 64.80% | 66.59% | 71.3 ms |
| LaFAM · AnyUp | 80.99% | 83.73% | 68.32% | 40.8 ms |

</div>

Method time includes the encoder and the map computation, with GPU synchronization. It excludes image loading, CPU scoring, and visualization. The reported median omits the first ten processed images per backbone to reduce startup effects. These timings describe this FP32, single-image run on an RTX 3060 Ti, not each method's maximum optimized throughput.

The accompanying results also contain paired AnyUp-minus-baseline differences and 95% intervals from resampling image pairs. Those intervals describe this sample; they do not cover different training runs or choices of LRP rules.

</details>

### Does the score reward the map—or its smoothness?

The ranking above raises a reasonable objection. Bilinear LaFAM starts with just 49 numbers. How can it beat a map that follows much finer image detail?

AnyUp also forms weighted mixtures of the original feature vectors in the [implementation used here](https://github.com/wimmerth/anyup/blob/351807a9c4287368732cc247f26c7c81c9139af4/anyup/layers/attention/chunked_attention.py). Its extra guidance comes from the image and the full feature vectors, which determine the mixing weights. More output pixels do not mean that the original encoder made more independent measurements.

A higher output resolution does not guarantee a better answer to the question being scored. Imagine a picture frame: its object pixels form an outline, but its bounding box also includes the empty center. A perfect outline leaves those empty-center pixels dark. The box metric counts them as missed foreground.

The controlled illustration below makes that mismatch visible. The object is a square outline whose true silhouette we know. Blurring its perfect map raises box AUROC from **71.9% to 98.0%**, while lowering silhouette AUROC from **100% to 89.3%**. The blur adds no evidence; it spreads the existing response into the box's empty center. The filled rectangle is an annotation-based illustration, not a model prediction.

<figure class="lf-figure" id="lf-metric-paradox">
<div class="lf-paths">
<div><strong>Exact object outline</strong><img src="{{ '/assets/img/posts/lafam/resolution/toy-0.png' | relative_url }}" alt="Exact square outline with an empty center"><span>Box AUROC: 71.9%<br>Silhouette AUROC: 100%</span></div>
<div><strong>The same outline, blurred</strong><img src="{{ '/assets/img/posts/lafam/resolution/toy-1.png' | relative_url }}" alt="Blurred outline spreading into its empty center"><span>Box AUROC: 98.0%<br>Silhouette AUROC: 89.3%</span></div>
<div><strong>Reduced to 7×7, then bilinear</strong><img src="{{ '/assets/img/posts/lafam/resolution/toy-2.png' | relative_url }}" alt="Coarse bilinear reconstruction of the outline"><span>Box AUROC: 65.1%<br>Silhouette AUROC: 92.8%</span></div>
<div><strong>Filled bounding rectangle</strong><img src="{{ '/assets/img/posts/lafam/resolution/toy-3.png' | relative_url }}" alt="Filled rectangle including the empty center"><span>Box AUROC: 100%<br>Silhouette AUROC: 89.3%</span></div>
</div>
<figcaption>A constructed example, not ImageNet results. The blur uses σ=32 pixels. A map can improve against the rectangle while becoming worse against the true object. This establishes that a ranking reversal is possible; it does not reveal the unknown true object contours in our ImageNet comparison.</figcaption>
</figure>

I then ran a separate sensitivity check on **200 fixed images**, taking every fifth entry of the original sample. For each method I applied the same Gaussian blurs, and also averaged its map back to $7\times7$ before bilinear enlargement. These settings were used to inspect sensitivity, not to select a new best-performing method.

<div class="lf-table-scroll" markdown="1">

| Activation-seeded LRP | Original | Blur σ=16 | Reduced to 7×7, then bilinear |
|---|---:|---:|---:|
| Supervised | 74.95% | 79.99% | 79.93% |
| SimCLR | 78.82% | 84.07% | 83.70% |
| SwAV | 74.27% | 79.14% | 78.83% |

</div>

Removing fine detail raises activation-seeded LRP's box AUROC by roughly five percentage points. The 7×7 reduction improves it too. That directly supports the concern that a coarse coverage metric can reward a smoother representation.

For AnyUp, applying the same σ=16 blur to both AnyUp and bilinear LaFAM makes their box AUROC values nearly coincide: the gap is at most 0.21 percentage points across the three encoders, compared with 0.35–1.09 points before smoothing on these same 200 images. Much of this small measured difference is therefore sensitive to spatial smoothing. This is a sensitivity check, not proof that their original maps are equally faithful or equally good at object boundaries.

There is an even simpler warning sign. A fixed central Gaussian—a soft blob that never reads the photograph or the encoder—scores **80.78% box AUROC** on the full 1,000-image sample. Its pointing score is **83.80%**. ImageNet framing and the center crop make central locations a strong baseline in this experiment. A high absolute localization score can therefore reflect the evaluation setup as well as image-specific information. This baseline exposes bias in the absolute scores; it does not, by itself, explain the ordering between methods.

Nearest-neighbor does beat bilinear on one of the original scores: **relevance mass inside boxes**, for all three encoders. Bilinear wins on box AUROC and pointing. Mass sums values; AUROC compares their ordering; pointing considers only maxima. Smoothing changes all three in different ways. There is no general rule that nearest or bilinear must win every metric.

### Why the LRP result needs a second look

The resolution mismatch explains part of LRP's measured disadvantage, but it is not the only variable. We have already seen a large change from replacing a uniform relevance seed with an activation-proportional seed.

The notebooks also explore starting from the **pooled embedding**. I checked that separately by placing an explicit global-average-pooling layer inside the LRP calculation, then averaging the embedding values per image. Its ordinary forward scalar equals the mean of the spatial features, but its modified backward pass includes an additional relevance-redistribution rule. The resulting map need not be the same.

Starting from the pooled embedding changes the supervised uniform-seed result from 67.37% to 71.49% box AUROC, but does not consistently improve the two SSL encoders. It is another meaningful definition choice, not an automatic fix.

The uniform spatial seed also assigns relevance to final features whose activation is zero. Between 44% and 92% of the final feature values are zero in this audit, depending on the encoder. An activation-proportional seed gives those features zero initial relevance. This distinction remains even when the mean is calculated separately for each image.

Excluding a 16-pixel zone around box edges does **not** remove LRP's disadvantage. That restricted score is defined on 169 images, equally for all methods. On those images, activation-seeded LRP is still about 8–14 points below bilinear LaFAM. This argues against blaming only the exact placement of rectangle boundaries. The fact that blurring helps is informative, but it does not explain every difference.

Finally, I checked the negative clipping itself. Keeping the signed channel mean changes uniform-seed LRP’s box AUROC by less than 0.15 percentage points, slightly lowering it for every encoder. Using absolute relevance changes the quantity being measured and does not close the gap either. Negative clipping is therefore not the main explanation here.

These checks support a narrower conclusion than “LRP is poor” or “the metric is wrong.” Fine relevance maps and coarse box coverage answer different questions, and the exact LRP definition matters. The box results alone cannot establish which method gives the most faithful explanation or the best object contours. We would need independent evidence for those claims.

The [audit code and results]({{ '/assets/downloads/lafam/metric-audit.zip' | relative_url }}) include all smoothing settings, the fixed center baseline, the pooled-target control, and per-image measurements. The original 1,000-image benchmark above is unchanged.


### What a rectangle cannot tell us

A box around a bicycle includes the spaces between its spokes. A map that traces the frame precisely can leave much of the box dark. A smooth map that illuminates the whole rectangle may score better, despite drawing a less convincing object boundary.

There is a second mismatch. ImageNet boxes annotate the target class, while these methods are label-free. Another visible object outside those boxes counts as background in this evaluation. Highlighting it can lower the score even when the feature response is meaningful.

Finally, a localization score is not a test of the explanation's faithfulness to the encoder. AnyUp can use the photograph to sharpen a feature map; LRP depends on its initialization and propagation rules. A better box score does not establish that either has recovered the unique explanation of the network.

The useful comparison therefore has two parts: inspect the real maps, and quantify a clearly stated task. Here the task is coarse localization against boxes. Evaluating fine contours would require suitable masks; evaluating attribution faithfulness would require a separate experiment on the model's behavior.


</details>

## The explanation was already there

The starting point is the encoder's own spatial representation. Averaging its feature channels gives a useful view of where learned patterns respond, with one forward pass and no target class or backward attribution. LaFAM makes that existing signal easy to inspect.

When the coarse grid hides spatial detail, AnyUp can refine the features using image guidance. That is an optional next step; the original model already supplied the activity we wanted to understand.

The examples here use ResNet-50 encoders. Later in this series we will examine how CNN and transformer mechanisms change the interpretation of a spatial feature. First, the next post asks how those features become a class decision, through **global average pooling and multiple-instance learning**.

<details class="lf-detail" markdown="1">
<summary>Reproducing this comparison</summary>

The experiment lives in the companion project's `lafam/` folder. `evaluate.py` makes and scores the maps; `report.py` generates the tables and visual assets. The README describes the data paths, crop, LRP seeds, checkpoint versions, and commands. The run saves its exact image IDs, per-image scores, exclusions, executed source, and file hashes.

[Download the code and measured results (ZIP)]({{ '/assets/downloads/lafam/lafam-box-comparison.zip' | relative_url }}), or read the [README]({{ '/assets/downloads/lafam/README.md' | relative_url }}), [per-image scores]({{ '/assets/downloads/lafam/scores.csv' | relative_url }}), and [paired differences with intervals]({{ '/assets/downloads/lafam/paired_differences.json' | relative_url }}). Model weights and ImageNet images are not included in the download.

The default is the 1,000-image sample used here. The code can evaluate all 50,000 validation images with `--per-class 50`; the full run is not claimed in this post.

</details>


## References

{%- capture references -%}
Karjauv, A., et al. | 2024 | LaFAM: Unsupervised Feature Attribution with Label-free Activation Maps;
Montavon, G., Samek, W., & Müller, K.-R. | 2018 | Methods for interpreting and understanding deep neural networks;
Adebayo, J., et al. | 2018 | Sanity checks for saliency maps;
Meehan, C., et al. | 2023 | Do ssl models have déjà vu? a case of unintended memorization in self-supervised learning;
{%- endcapture -%}

{% include reference.html ref=references %}
