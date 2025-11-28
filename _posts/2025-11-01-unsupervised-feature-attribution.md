---
layout: post
title: Unsupervised Feature Attribution for Foundation Models
date: 2025-11-01 00:00:00 +0200
tags: XAI, CNN, Foundation Models, Self-Supervised Learning
description: How activation maps and relevance propagation can shed light on deep models without any labels.
---

Foundation models trained without labels can learn a lot, but understanding what they actually see is another story. Convolutional networks, though, hide a clue: their activation maps quietly reveal where the model is looking. No labels, no gradients, just the network explaining itself. The only problem is that its vision is still a bit blurry… for now.

<!--more-->

This journey started with my attempt to use Generative Adversarial Networks (GANs) as feature extractors for medical images. GANs can synthesize high-quality images from random noise and control the generation process via latent variables, but they are not inherently invertible. I tried to address this limitation, but it didn’t work well (see my earlier [post](/2023/01/06/turning-stylegan-into-a-latent-feature-extractor.html) for details). Debugging that encoder-based GAN quickly turned into an exercise in frustration, exposing how few tools exist to reveal what such models actually learn. That limitation led to a broader question -- how can we explain models that operate without labels? This question is central to self-supervised learning (SSL), where models learn latent representations directly from unlabeled data. How can we see what such models focus on? That question brings us to LaFAM [<a href="#karjauv2024lafam" data-ref="karjauv2024lafam">Karjauv et al.</a>], a straightforward but effective method for unsupervised feature attribution.

This post explores how LaFAM provides quick saliency maps for CNNs in an unsupervised setting, born out of the need to interpret SSL models. We'll also push further, extending LaFAM with LRP to gain higher-resolution insights. Along the way, we'll maintain a critical lens: Are our evaluation metrics actually fair? What limitations do these methods have?

<!-- By the end, we’ll see not only where the network looks, but also reflect on the perennial XAI question of “so what?” (or rather what is it seeing there?). -->

<!-- (If you're just joining, no worries – each part stands alone. But know that Part 1 dealt with GANs and encoder limitations, and Part 3 will tackle Multiple Instance Learning. Now let's dive in!) -->

## Self-Supervised Learning Meets XAI

Self-Supervised Learning (SSL) has emerged as a way for models to learn useful representations without manual labels. Vision models like SimCLR and DINO can train on millions of images by solving proxy tasks (e.g., contrasting different augmented views of the same image) and then be fine-tuned for actual tasks. SSL models are often called foundation models for their broad adaptability. Yet, the absence of labels makes it difficult to verify whether the learned features are actually relevant for a given downstream task. Moreover, when trained with proxy tasks such as random cropping, a model may unintentionally associate irrelevant features without us realizing, and evaluating it is non-trivial [<a href="#meehan2023do" data-ref="meehan2023do">Meehan et al.</a>].

Explainable AI (XAI) offers ways to probe a model’s reasoning, for example, by producing saliency maps that highlight important regions of an input. Traditional XAI methods, though, are designed for supervised models by attributing input importance for a specific class. Methods like Grad-CAM and occlusion-based methods (e.g., RISE) require a so-called score function. This function takes a target class as input to guide the attribution. This poses a problem for SSL, as there are no explicit labels to explain.

There have been attempts to adapt XAI to label-free models. One such attempt is RELAX (Representation Learning Explainability) which extended an occlusion method (RISE). The key idea is clever. SSL models output embeddings that are unitless, meaning that each value does not refer to any particular feature. Since we don't know what a target embedding should look like, the authors propose to first create a reference embedding from the original input and extract embeddings from occluded inputs. We can then define a score function that measures cosine similarity between the reference embedding and the occluded ones and use it to attribute the most salient features.

However, RELAX ended up being computationally expensive as it needs many forward passes with different masks, and it often produced very noisy maps. Moreover, using random patch occlusions can introduce unnatural artifacts, leading the model to react strangely to a big gray patch that it would never see during normal operation.

## Label-Free Activation Maps

<!-- An important advantage of CNNs is that their spatial feature maps preserve the structure of the input image. As layers stack, each convolution processes a local region of the previous feature map, which causes receptive fields to expand with depth. By the final convolutional layer, neurons cover enough of the image to encode class-specific signals while still retaining coarse spatial layout. -->

A key strength of CNNs is that their activation maps maintain a connection between detected patterns and their positions in the input image. This property makes the models inherently more explainable and underpins the success of Class Activation Map (CAM) methods. By weighting the maps in the final convolutional layer according to their contribution to a target class, these methods can localize the image regions most relevant in supervised settings.

<details><summary>What is an activation map?</summary>
<p>
Each convolutional layer takes an input and produces activation maps (also called feature maps) that record activations of specific visual patterns across the image. The first layer processes raw pixels and responds to simple local features such as edges or color contrasts. Each subsequent layer takes the feature map from the previous one and combines these basic patterns into more complex and abstract representations, like textures or object parts. The area a neuron responds to is called its receptive field. As the network goes deeper, receptive fields grow as a result of pooling operations or convolution strides, which reduce the spatial size of feature maps. This lets deeper neurons capture a larger portion of the image while still preserving coarse spatial relationships.
</p>
</details>

<!-- In other words, each channel in a conv layer can be seen as a spatial map of how strongly that feature is present across the image. For example, a deep CNN might have a channel that activates on “dog faces” and another on “fur texture,” each producing a 2D map of where that feature is present in the image.  -->

But what if we don’t have a class? The answer is simple -- we don't need it. We can simply average all the activation maps at the last convolutional layer to get a generic saliency map. This label-free map doesn’t focus on any one class. It treats every learned feature as equally interesting, highlighting regions that strongly activate any of the high-level features in that layer. Essentially, it’s a visualization of “where the network is looking” in a class-agnostic sense.

LaFAM (Label-free Activation Map) [<a href="#karjauv2024lafam" data-ref="karjauv2024lafam">Karjauv et al.</a>] evaluates this approach systematically. The method is astonishingly simple yet effective. It outperforms RELAX and even stands up well against Grad-CAM.

### Qualitative Comparison

<figure>
<img src="/assets/img/lafam/pascal_voc_2012_results.svg" alt="LaFAM vs. Grad-CAM vs. RELAX" />
  <figcaption>Comparison on PASCAL VOC 2012. LaFAM produces maps similar to Grad-CAM but remains robust when the model misclassifies (first row).</figcaption>
</figure>

In the figure above we see this comparison in action. This figure compares three attribution methods: LaFAM, RELAX, and Grad-CAM as a baseline. LaFAM produces very similar saliency maps to those produced by Grad-CAM in supervised models (i.e., applied to the same supervised model).

In the first row, the model actually mispredicted the ImageNet class, so Grad-CAM dutifully highlighted an irrelevant region associated with the wrong class. This illustrates a subtle strength--by not being yoked to the top-1 predicted label, simply averaging the last CNN layer won’t completely go off the rails when the model’s prediction is off. It shows everything the model found salient, not just what influenced the (possibly incorrect) class choice. For SSL models, LaFAM is notably less noisy than RELAX’s outputs.

The next figure demonstrates that LaFAM, being class-agnostic, highlights multiple objects in the image (e.g., both dogs), whereas Grad-CAM focuses on just one object tied to the predicted class. This multi-object sensitivity is a useful trait in many scenarios, as real-world images almost always contain several relevant items.

![LaFAM for Multiple Objects](/assets/img/lafam/pascal_voc_2012_2_classes.svg){: .center-image }

Let's look closer at the model misprediction cases. For Grad-CAM we visualize the saliency map for the (wrong) predicted class. As a result, Grad-CAM highlights irrelevant regions, while LaFAM attributes all learned features. This again underscores the advantage of being label-free: LaFAM reflects what the model finds salient overall, not just what it thinks is the “correct” class.

![Grad-CAM for Missclussification Case](/assets/img/lafam/imgnet_missclf_short.svg){: .center-image }

LaFAM clearly highlights true objects, suggesting that these objects strongly activate multiple channels in the final conv layer. However, the prediction is wrong. The reason could be that the last fully connected layer puts more weight on specific features that mislead the final decision, even though many other features correctly identify the object.

## What Do the Numbers Say?

To move beyond visual inspection, Karjauv & Albayrak tested LaFAM on standard datasets (ImageNet-S and PASCAL VOC 2012) using segmentation masks as ground truth. The logic is simple: if the saliency map highlights the pixels that actually belong to the object, the attribution is accurate.

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

Karjauv & Albayrak put LaFAM to the test on standard datasets to see how well these label-free maps coincide with actual objects in the images[0]. Since we don’t have class labels to evaluate “correctness” of an explanation, they leveraged datasets with segmentation masks (ImageNet-S and PASCAL VOC) and a suite of evaluation metrics from the Quantus XAI evaluation framework[0]. Essentially, they treated it as a localization task: if a saliency map highlights the region of the true object (per segmentation mask), it’s doing a good job. They compared LaFAM against RELAX (the prior SSL method) using SSL models (SimCLR and SwAV on ResNet50 backbones), and also compared LaFAM against Grad-CAM for a fully supervised ResNet50 classifier as a sanity check[0]. All saliency maps were upscaled to image size for fairness, and they computed metrics like:

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

### Analyzing the Results

The results tell us a story that goes deeper than just comparing two methods. The high performance of LaFAM confirms that CNNs inherently develop detectors for semantic objects, even without explicit supervision. Remember, LaFAM is simply averaging all activations in the final layer. The fact that this naive average aligns so perfectly with ground-truth objects implies that the most active features in the network are naturally the semantic objects. The network has learned to ignore the background noise on its own.

Beyond its application in SSL, LaFAM offers a unique lens for debugging supervised models by removing the "tunnel vision" inherent in class-specific methods. While tools like Grad-CAM essentially tell the model to ignore everything except the target class, LaFAM reveals the quality of the learned representation as a whole. This holistic view is critical for diagnostics: it allows us to distinguish between a model that is "blind" (failing to detect the object entirely) and one that is "confused" (seeing the object clearly but misclassifying it due to the final decision layer). By highlighting everything the generic feature extractor considers important—rather than just what supports a specific decision—we can more easily spot when a model is relying on background shortcuts or ignoring secondary objects in complex scenes.

Besides the application in SSL, LaFAM can also be useful for debugging supervised models. CAM-based methods use the class label to filter the view—we essentially tell the model, ignoring everything else. But sometimes we want to verify the quality of the learned representation as a whole. We need to know everything the model considers important: If a supervised model is misclassifying an image, averaged convolution can reveal whether the model is focusing on the right object or getting distracted by irrelevant features. This can help diagnose issues like dataset bias or spurious correlations.

The comparison with RELAX highlights this difference. RELAX struggles (low Sparseness) because it relies on perturbing the input to see what changes the embedding. LaFAM, by contrast, directly maps the presence of the learned features. The huge gap in performance suggests that the information was already there in the activation maps—we just needed to look at it directly rather than poking it with occlusion masks.

In the Supervised setting, LaFAM is competitive with Grad-CAM, but with a key distinction. Grad-CAM wins on "Sparseness" because it artificially suppresses features belonging to other classes. LaFAM includes them. While this hurts the sparseness score, it arguably provides a more honest view of the model's state: if the model sees a Cat and a Dog, LaFAM shows you both.

## Extending LaFAM: The Quest for Resolution

While LaFAM is fast and effective, it has one inherent limitation: resolution. It relies on the final convolutional layer, which in a ResNet50 is a coarse $7 \times 7$ grid. Upsampling this to the image size results in "blobby" heatmaps that show the general location of an object but miss the fine contours.

The paper suggests a potential path forward: combining LaFAM with Layer-wise Relevance Propagation (LRP).

LRP is a technique that normally starts with a class score and propagates it backward through the network, distributing "relevance" to neurons that contributed to that score. In a label-free context, we don't have a class score. However, we can treat the LaFAM map itself as the starting signal. By redistributing the coarse activation values from the top layer back down to the pixel level using LRP rules, we might theoretically recover fine-grained details—like the sharp edges of a cat’s ear—that the coarse map smooths over.

While this extension is still exploratory, it represents an exciting direction. It would bridge the gap between the semantic richness of deep layers and the spatial precision of shallow layers, all without needing a single label.

## Are Our Evaluation Metrics Actually Fair?

Looking at the tables, you might notice that Grad-CAM often beats LaFAM on metrics like **Sparseness** or **Top-K Intersection**. But we should pause and ask: do these metrics measure what we *actually* want?

Most standard XAI metrics, including the Pointing Game, implicitly assume that there is **one** correct object in the image—the one matching the ground-truth label. If a picture contains a dog and a cat, but the label is "dog," a method that highlights both will be penalized by these metrics.

LaFAM is, by design, class-agnostic. It highlights *everything* the network finds salient. As we saw in the qualitative examples, this often leads to LaFAM highlighting multiple valid objects in a scene. Paradoxically, this makes it "worse" according to standard metrics, even though it might be providing a more honest view of the model's internal representation.

This suggests that while metrics are useful for comparison, they are not neutral arbiters of truth. They encode specific assumptions—sparsity, single-object focus—that may not align with the goals of unsupervised learning.

## Limitations and Key Takeaways

LaFAM is not a silver bullet. It is designed specifically for CNN architectures where spatial information is preserved in feature maps; applying it to Vision Transformers (ViTs) would require a different approach (perhaps using attention heads). Furthermore, without the LRP extension, the resolution remains coarse, which might not be precise enough for tasks like medical segmentation where pixel-perfect boundaries matter.

However, for practitioners working with SSL foundation models, LaFAM offers a practical, plug-and-play tool. It provides an immediate sanity check: *Is my model looking at the object, or did it just memorize the background grass?*

**Key Takeaways:**

*   **LaFAM is fast and label-free:** It turns CNN activation maps into saliency maps with a single forward pass, making it ideal for Self-Supervised Learning.
*   **Beats the alternative:** It produces cleaner, less noisy maps than occlusion-based methods like RELAX, without the heavy computational cost.
*   **Robust to errors:** In supervised settings, LaFAM remains stable even when the model predicts the wrong class, often highlighting the true object when Grad-CAM is misled.
*   **Metrics are tricky:** Standard evaluation metrics penalize methods that highlight multiple objects, potentially undervaluing the rich, multi-faceted representations learned by SSL models.

By accepting that we don't always need a label to explain a model, we open the door to understanding the "general purpose" vision of foundation models—blurs, blobs, and all.

## References

{%- capture references -%}
Karjauv, A., et al. | 2024 | LaFAM: Label-Free Activation Mapping for Interpreting Self-Supervised Vision Models;
Montavon, G., Samek, W., & Müller, K.-R. | 2018 | Methods for interpreting and understanding deep neural networks;
Adebayo, J., et al. | 2018 | Sanity checks for saliency maps;
Meehan, C., et al. | 2023 | Do ssl models have déjà vu? a case of unintended memorization in self-supervised learning;
{%- endcapture -%}

{% include reference.html ref=references %}
