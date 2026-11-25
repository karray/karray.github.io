---
layout: post
title: "Your Classifier Has Always Been a Multiple-Instance Learner"
description: "Why standard image classifiers already recognise multiple objects in space, how certain head designs break that signal, and what comes for free once you read before pooling."
styles:
  - /assets/css/gap-mil.css
tags: XAI, Image Classification, Multiple-Instance Learning, CAM, CNN, ViT
---

We keep treating standard image classifiers as single-output models, leaving their built-in spatial capabilities untapped. Bypassing global average pooling in a pretrained classifier turns a single global prediction into a map of local class scores for free. Anyone working with vision models can use this spatial map right away to audit model failures and examine how networks handle unlabelled background. This post shows why the maths works, how convolutional networks and vision transformers differ under the hood, and what these hidden scores reveal in multi-object scenes.

<!--more-->

Step outside with a camera and snap a photo of a dog on a busy street or a coffee cup on a café table. Real-world images almost never isolate a single subject against a blank studio wall. Large parts of every frame consist of unlabelled background with no dedicated category, often alongside other objects. Standard image classifiers are nevertheless built to return just one class for the entire scene. That single-label design exists for practical reasons, as annotating data for dedicated object detectors is expensive. Assigning one label per image is far cheaper than manually outlining every item across thousands of pictures, allowing researchers to scale datasets into the millions.

Even when trained with a single label per image, a vision network does not process the scene as one undifferentiated block. Whether an architecture relies on convolutional layers or transformer blocks, it transforms an input image into a coarse spatial feature grid. Each cell in that grid holds a vector of numbers describing what the network has detected in the corresponding region of the scene. In an earlier post on [label-free feature attribution]({% post_url 2026-10-01-unsupervised-feature-attribution %}), we averaged each cell across its feature channels to reveal where prominent features respond without using any class labels.

{% include posts/label-free-lrp-stack.html %}

To see how those spatial features turn into a category label, it helps to view a vision classifier as a feature encoder and a linear classification head. A typical encoder transforms a $224 \times 224$ image into a $7 \times 7$ grid of $C$-dimensional feature vectors. The linear head expects a single $C$-dimensional vector and maps it to $K$ class scores called logits, and the highest-scoring category becomes the final image-level prediction. To bridge these two stages, most modern classifiers place global average pooling (GAP) in between, averaging each channel across all 49 cells to collapse the grid into one summary embedding.

<details markdown="1">
<summary>Why GAP is a default pooling in modern models</summary>
Early deep convolutional networks, such as AlexNet and VGG, converted spatial grids into predictions by flattening the entire feature tensor into one long vector and feeding it into parameter-heavy fully connected layers. With millions of weights tied to specific grid coordinates, a classification head had to learn the same object-to-feature associations independently for every location and could easily memorise training pictures instead of learning general visual concepts. Keeping those networks from overfitting forced engineers to rely heavily on aggressive regularisation such as dropout. Flattening also locked models to a fixed spatial layout, breaking whenever an input image arrived at a different resolution.

GAP addresses these limitations by eliminating learnable parameters at the transition to the classification head. Averaging each feature channel across the grid converts a spatial map of local responses into a single score measuring how strongly a visual pattern appears across the scene. If a channel responds to fur texture or bicycle spokes, its spatial mean captures the overall presence of that feature no matter where the subject appears in the frame. Without a downstream dense layer to recombine scattered signals, individual channels are forced to encode meaningful semantic concepts directly. Computing a mean over whatever grid dimensions the encoder outputs also frees the network from a fixed input resolution while building translation invariance into the representation. The trade-off is that GAP discards relative spatial geometry, treating the picture as an unordered collection of local detections. That geometric blind spot is harmless when assigning a single category label to an image, yet it makes GAP poorly suited for robotics and embodied agents that need exact coordinates to grasp objects or navigate around obstacles.
</details>

Combining spatial averaging with a linear classifier gives us a useful mathematical property. In ordinary arithmetic, averaging a list of numbers and multiplying by a weight gives the exact same result as multiplying each number by that weight first and averaging afterwards. Adding a constant bias behaves the same way. Since GAP is an average and a linear head is a weighted sum plus a bias, the two operations can be swapped. Once a model is trained, we can skip pooling altogether and apply the trained classification head directly to every cell in the feature grid. Applying the classifier pointwise replaces the single global output with a $7 \times 7$ map of local class scores known as a dense readout. Taking the spatial average of those local scores recovers the original image-level logits exactly.

<details markdown="1">
<summary>Math behind it</summary>

Consider a model $M = f \circ \text{GAP} \circ E$, where $E$ is encoder and $f$ is a linear classification head. Given an input image $x$, the encoder produces a spatial feature tensor $F = E(x) \in \mathbb{R}^{C \times H \times W}$, where $C$ is the number of feature channels and $(H, W)$ are the grid dimensions. Flattening this grid gives $N = H \times W$ cells corresponding to distinct patches in $x$, with each cell holding a $C$-dimensional feature vector $h_i \in \mathbb{R}^C$ for $i = 1, \dots, N$. The pooling operator collapses these $N$ vectors into a single mean vector $\bar{h} = \text{GAP}(F) = \frac{1}{N} \sum_{i=1}^{N} h_i$.

For any target category $c$, a linear classification head $f_c(h) = w_c^\top h + b_c$ uses a learned weight vector $w_c \in \mathbb{R}^C$ and scalar bias $b_c \in \mathbb{R}$ to compute the global logit $z_c$. Since multiplication and addition distribute across summation, we can bypass $\text{GAP}$ and move the linear head inside the average.

$$
z_c = \underbrace{w_c^\top \left( \frac{1}{N} \sum_{i=1}^{N} h_i \right) + b_c}_{f_c(\bar{h})}
    = \frac{1}{N} \sum_{i=1}^{N} \underbrace{\left( w_c^\top h_i + b_c \right)}_{f_c(h_i) = Z_{c,i}}
$$

Applying the trained head $f$ independently to each cell's feature vector $h_i$ yields the local logit $Z_{c,i}$ for class $c$ at grid cell $i$. Reshaping these scores across all $K$ categories back onto the $H \times W$ grid produces the dense readout $Z \in \mathbb{R}^{K \times H \times W}$, whose spatial average equals the global logit vector $z$. Taking $\arg\max_c Z_{c,i}$ at each cell then converts those continuous local scores into a grid of top-1 patch predictions.

</details>

That equivalence reveals how standard classifiers learn from complex scenes. During training, the network implicitly scores every cell across the spatial grid, yet the dataset provides only a single image-level label for the whole input. In machine learning, training on an unlabelled collection of items with a single class label is known as multiple-instance learning (MIL). Each image acts as a bag, and its grid cells serve as individual instances that remain unlabelled throughout training. Treating those cells as strictly independent instances is only a working approximation, as overlapping convolutional receptive fields and transformer self-attention share context across neighbouring regions. Even so, the linear equivalence holds exactly, meaning every GAP classifier with a linear head already functions as a mean-aggregation MIL model.

<details  markdown="1">
<summary>Where does multiple-instance learning come from?</summary>
Researchers originally formulated MIL in the late 1990s for drug discovery, where screening tests indicate whether a candidate molecule is active without revealing which of its possible three-dimensional conformations caused the reaction [<a href="#dietterich1997solving" data-ref="dietterich1997solving">Dietterich et al.</a>, <a href="#maron1998a" data-ref="maron1998a">Maron and Lozano-Pérez</a>]. In this setup, the candidate molecule acts as a bag, and each individual shape it can adopt counts as an instance inside that bag. Training labels apply only to the bag as a whole while the individual instances remain unlabelled.

The framework later became essential in digital pathology, where gigapixel tissue scans are far too large for GPU memory to process in a single pass and signs of disease often hide in a tiny fraction of the slide. Marking every tumour cell by hand is impractical at scale, yet clinical archives already pair each slide with a single patient diagnosis. Slicing the scan into small tiles lets a model process the slide piece by piece as a bag of unlabelled tissue patches.
</details>

<details markdown="1">
<summary>What happens when Softmax or LayerNorm sits at the end?</summary>

Two practical operations can break this exact decomposition if you apply them in the wrong place. First, never average probabilities. Classifiers turn logits into probabilities using the softmax function, which exponentiates scores and divides by their sum. Because softmax is nonlinear, applying softmax inside each cell and averaging those probabilities changes both the numbers and sometimes the winning class. Try ticking the *Apply Softmax before averaging* box in Figure 1 to watch the two routes diverge.

Second, some architectures place a LayerNorm layer *after* global average pooling and *before* the linear head. ConvNeXt-B does this, and MaxViT-B adds a $\tanh$ layer as well. If you ignore that LayerNorm or normalise each cell independently, the spatial mean can drift away from the actual image logits by dozens of points. To keep the decomposition exact in ConvNeXt, each cell $h_i$ is centred by its own channel mean $\mu_i$ and divided by the *pooled* vector's spread $\sigma(\bar{h})$ before the linear head, preserving the exact sum to $10^{-6}$ [<a href="#karjauv2026rethinking" data-ref="karjauv2026rethinking">Karjauv</a>].

</details>

## A built-in diagnostic toolkit

Keeping the MIL structure intact does more than expose secondary objects. Every spatial vector and its local logit vector become a zero-cost diagnostic lens for understanding how a model arrives at its predictions, where it cheats, and even when our own evaluation tools mislead us.

### Small objects are outvoted

Viewing image classification as mean-aggregation MIL immediately explains why capable models appear to fail on complex or cluttered images. When a small object occupies only a handful of grid cells while distracting surroundings or a second large object occupy the rest, strong local evidence gets diluted by the average. Even if all six cells covering a teddy bear vote enthusiastically for *teddy*, forty-three surrounding cells leaning toward *syringe*, *ashcan*, or *bottle* can easily outvote them in the final sum.

On *Pets indoors*, ResNet-50 and Swin-B both output *window screen* globally and leave the cat out of their top-5 predictions, yet silencing the window cells reveals both the *golden retriever* and the *Egyptian cat* waiting underneath. On *ImageNet-A*, a benchmark of natural images famous for crushing ImageNet models down to 7.2% on ResNet-50 [<a href="#hendrycks2021natural" data-ref="hendrycks2021natural">Hendrycks et al.</a>], the dense readout tells a completely different story. Across all 7,500 ImageNet-A images, the true class still wins at least one grid cell 30.6% to 56.8% of the time at top-1, and 57.5% to 80.0% at top-5. In many of those failures, perception worked fine, and mean aggregation lost the vote.

Meanwhile, across the 50,000 ImageNet validation images and their multi-label ReaL re-annotations, every tested GAP classifier detects the ground-truth class inside its foreground bounding box on 93.7% to 95.2% of images and recovers 89.5% to 92.0% of valid ReaL labels directly on the grid [<a href="#karjauv2026rethinking" data-ref="karjauv2026rethinking">Karjauv</a>]. The information we kept trying to patch with top-5 tolerance and expensive human re-labelling was sitting inside the pre-pooling tensor the whole time.

### Shortcuts and correlations

tk: fish/person, Bearskin/uniform...

### Revealing adversarial attacks

Watching an adversarial attack flip the bag. Standard adversarial evaluations check whether a tiny perturbation fools the global prediction [<a href="#madry2018towards" data-ref="madry2018towards">Madry et al.</a>], leaving the spatial mechanics hidden. Scrub through the twenty Projected Gradient Descent (PGD) steps in Panel 2, pushing ResNet-50's prediction on a bulldog and cat from *bull mastiff* toward *sombrero*. Rather than shifting the entire image uniformly, the attack gradually corrupts local votes until just 5 of 49 cells shouting loudly enough for *sombrero* tip the pooled average at step 8, reaching 12 of 49 cells by step 20.


## Classifiers are implicit MIL

A sceptical readers may raise a fair objection at this point. Showing that a pretrained ImageNet model lights up on multiple objects is neat, yet ImageNet consists mostly of centred single objects, and pretrained models might simply pick up local textures by chance. How could training with one label per image teach a network to localise several unrelated objects reliably?

During training, cross-entropy loss pushes the pooled bag logits toward whichever single label was annotated, which sends the exact same gradient to every spatial cell before pooling. On a single picture containing a dog and a bicycle labelled *dog*, the network receives no hint telling it which pixels belong to the dog and which belong to the bicycle. Across hundreds of thousands of images, however, objects recombine. Dogs appear beside bicycles, sofas, grass, and cars, while bicycles appear beside walls, riders, and dogs. Over many updates, features matching dogs correlate consistently with the *dog* label, while features matching bicycles correlate with images labelled *bicycle*, allowing individual instances to specialise even though no training image ever carries two labels.

To prove this credit-assignment mechanism cleanly without relying on pretrained weights or single-object datasets, we can train a model on a controlled synthetic multi-object benchmark (SynMIL) [<a href="#karjauv2026rethinking" data-ref="karjauv2026rethinking">Karjauv</a>]. Each image contains three to five coloured geometric symbols drawn from twenty distinct shapes, placed at random positions, scales, and rotations. Crucially, the training set attaches only one randomly chosen visible symbol as the image label, leaving the other two to four shapes completely unlabelled.

Think about the mathematical ceiling on image-level accuracy in that game. Even an oracle with perfect vision cannot tell which of the $n \in \{3, 4, 5\}$ visible objects the annotator happened to pick for a given test image, so its best possible single-label accuracy is $\mathbb{E}[1/n] \approx 26.1\%$. When we train a ResNet-18 and a GAP-pooled ViT-S/32 from random initialisation on this ambiguous dataset, both hit that theoretical image-level ceiling. Judged by standard validation accuracy, they look terrible.

Open up the dense readout before pooling, however, and ResNet-18 predicts a visible object at the image level 100.0% of the time and correctly classifies 94.1% of all foreground grid cells across the twenty symbol classes. Even with only one random label per bag and zero bounding boxes, mean-pooling MIL forces the encoder to become an accurate multi-object detector.







## CNNs vs. ViTs

tk: local receptive fields vs. self-attention contextualisation

The same lens clarifies why local cells in a scene sometimes name a close relative of the true object. Look closely at the legs of the ox in the *Six objects* collage under Swin-B or ResNet-50, and you will see two cells predicting *water buffalo*. A single $32 \times 32$ pixel square covering a dark hoof and lower leg genuinely resembles both bovids, so that cell is reporting evidence that is accurate at a coarser level of the WordNet label hierarchy even when it cannot resolve the fine species alone. Deciding whether the animal is an ox or a water buffalo is a bag-level question settled across the whole animal, including the horns and coat, not a mistake by an isolated leg cell.

Separating context spreading from shortcuts under occlusion. Because instances can communicate through receptive fields or self-attention, a cell sitting on grass or sky can vote *warthog* for two completely different reasons. Either the network learned a spurious shortcut linking savannah grass to warthogs, or attention carried news about the warthog out into the grass. Covering the annotated foreground box tests the two directly. In Swin-B and MaxViT-B, background cells predict the true ImageNet class on 89.2% and 90.1% of validation images when the object is visible, then plummet to 35.8% and 37.9% once the object is occluded. Try toggling *Warthog* in Panel 1 to watch Swin-B's forty-nine *warthog* votes vanish without the animal. Contrast that with *Tench*, where removing the fish cleanly from the angler's hands leaves 43 of 49 cells in Swin-B still voting *tench* because the model relies on the angler as a shortcut [<a href="#geirhos2020shortcut" data-ref="geirhos2020shortcut">Geirhos et al.</a>]. Meanwhile, for classes like *web site*, *basketball*, and *bearskin*, surrounding court players or uniforms keep background activation above 95% across architectures even when the target object is masked.

note: Catching evaluation artefacts in occlusion tests. Occluding pixels with a flat grey rectangle is itself an out-of-distribution edit, and dense readouts catch when that edit fools the model. In ImageNet validation image `15750` (*garden spider*), ResNet-50 predicts *garden spider* on the spider and *spider web* across the surrounding silk. Masking the spider with a grey box leaves the image prediction stuck on *garden spider*, which might tempt you to conclude that the web alone triggered a spider shortcut. Yet when we remove the spider cleanly using generative inpainting, ResNet-50 immediately predicts *spider web* across 39 of 49 cells and drops *garden spider* to zero. Put that same grey occlusion box back onto the *already spider-free* web, and 12 adjacent web cells flip right back to *garden spider*! Overlapping convolutional receptive fields react to the sharp unnatural box boundary, a hidden evaluation bug you would never spot from a global accuracy score.

## Practical recommendations and what comes next

Once you recognise that a classifier with GAP and a linear head is an exact MIL model, two popular architectural habits suddenly look like self-inflicted wounds. Both leave image-level accuracy almost unchanged, yet both silently sever the exact link between local visual evidence and the final decision.

The first trap is the `[CLS]` token in Vision Transformers. When ViT was introduced [<a href="#dosovitskiy2021an" data-ref="dosovitskiy2021an">Dosovitskiy et al.</a>], the authors copied the special classification token from BERT in natural language processing, prepending an extra blank vector to gather information from the patches so the head could read only that vector at the end. Yet in their own ablation appendix (Appendix D.3), the authors noted that an initial poor score with global average pooling had nothing to do with the `[CLS]` token or with GAP itself, and simply reflected a mismatched learning rate. Properly tuned, GAP performs just as well as `[CLS]`, while keeping every patch token inside the exact bag sum. When you apply a `[CLS]`-trained head to patch tokens as an informal probe, as shown in Figure 1 with `ViT-B/16 [CLS]`, the spatial average misses the actual image logits by up to 9.47 logits across ImageNet-1K because the classifier never trained on the patches directly [<a href="#karjauv2026rethinking" data-ref="karjauv2026rethinking">Karjauv</a>].

The second trap is stacking a multi-layer perceptron (MLP) head after pooling instead of a single linear layer. Back in 2016, the original CAM paper already had to strip away AlexNet's and VGG's multi-layer fully connected heads and replace them with GAP plus a single linear projection so spatial scores could be read [<a href="#zhou2016learning" data-ref="zhou2016learning">Zhou et al.</a>]. Years later, in self-supervised learning, the SimCLR paper examined why placing a nonlinear MLP projection head after the pooled representation changed what survived at the output (Section 4.2 and Appendix B.4) [<a href="#chen2020a" data-ref="chen2020a">Chen et al.</a>]. A nonlinear MLP acts as an information bottleneck that combines pooled features nonlinearly, discarding spatial and transformation details that live cleanly before the head.

To measure that damage directly in supervised classifiers, we can train pairs of ResNet-18 and GAP ViT-S/32 models on ImageNet-1K, swapping only the final single linear layer for a three-layer MLP with ReLU activations [<a href="#karjauv2026rethinking" data-ref="karjauv2026rethinking">Karjauv</a>]. For ResNet-18, image-level top-1 accuracy nudged up slightly from 70.5% to 71.4%, looking completely healthy on a standard leaderboard. Underneath, however, foreground cell localisation collapsed from 52.8% down to 16.3%. Because the MLP only sees the globally pooled vector during training, it learns to rely on nonlinear interactions between features scattered across the entire image, features that rarely coexist inside a single spatial cell before pooling. (In ViT-S/32, self-attention already mixes features across tokens before pooling, softening the drop from 49.7% to 47.4%, yet the exact bag-sum identity is still lost.)

The architectural takeaway costs nothing to adopt. Unless a downstream objective forces you to isolate a bottleneck, average spatial features with GAP and classify them with a single linear layer.

Treating standard classifiers as the multiple-instance learners they already are changes how you design, train, and inspect vision models, without adding a single extra parameter or training loss.

- Pair GAP with a single linear head. In both CNNs and Vision Transformers, skip the `[CLS]` token and avoid nonlinear MLP heads when you want faithful spatial attribution. You keep full image-level accuracy while retaining an exact, additive decomposition over spatial instances.
- Train on pooled embeddings with cheap labels, then unpool at test time. Single-label supervision over multi-object scenes still teaches accurate local class detectors across the dataset, as SynMIL shows cleanly. Never discard spatial grids before checking what they already separate.
- Inspect the dense readout before blaming the encoder. Whenever a classifier fails on a cluttered image or an adversarial benchmark like ImageNet-A, compare the global winner against the local cell grid to tell perception failures apart from simple pooling dilution, shortcuts, and mask artefacts.

One subtle puzzle still remains. Standard classifiers have no background or abstain option inside their thousand-class head, which means every single empty patch of sky, carpet, or grass is forced to cast a vote for *some* foreground object class. Simply thresholding low-confidence cells does not fix this cleanly, because sweeping confidence or prototype thresholds reduces background false alarms only by sacrificing foreground localisation along with them [<a href="#karjauv2026rethinking" data-ref="karjauv2026rethinking">Karjauv</a>]. How CNNs keep background cells quiet by shrinking their feature vectors while Vision Transformers use attention to fill background tokens with the object's identity is the geometry we will unpack in the next post.

## References

{%- capture references -%}
Dietterich, T. G., Lathrop, R. H., & Lozano-Pérez, T. | 1997 | Solving the Multiple Instance Problem with Axis-Parallel Rectangles;
Maron, O., & Lozano-Pérez, T. | 1998 | A Framework for Multiple-Instance Learning;
Lin, M., Chen, Q., & Yan, S. | 2014 | Network In Network;
Oquab, M., Bottou, L., Laptev, I., & Sivic, J. | 2015 | Is Object Localization for Free? Weakly-Supervised Learning with Convolutional Neural Networks;
Russakovsky, O., et al. | 2015 | ImageNet Large Scale Visual Recognition Challenge;
Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., & Torralba, A. | 2016 | Learning Deep Features for Discriminative Localization;
Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. | 2018 | Towards Deep Learning Models Resistant to Adversarial Attacks;
Beyer, L., Hénaff, O. J., Kolesnikov, A., Zhai, X., & van den Oord, A. | 2020 | Are We Done with ImageNet?;
Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. | 2020 | A Simple Framework for Contrastive Learning of Visual Representations;
Cohen, J. P., Morrison, P., Dao, L., Roth, K., Duong, T. Q., & Ghassemi, M. | 2020 | COVID-19 Image Data Collection: Prospective Predictions Are the Future;
Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. | 2020 | Shortcut Learning in Deep Neural Networks;
Dosovitskiy, A., et al. | 2021 | An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale;
Hendrycks, D., et al. | 2021 | Natural Adversarial Examples;
Raghu, M., Unterthiner, T., Kornblith, S., Zhang, C., & Dosovitskiy, A. | 2021 | Do Vision Transformers See Like Convolutional Neural Networks?;
Shao, Z., Bian, H., Chen, Y., Wang, Y., Zhang, J., Ji, X., & Zhang, Y. | 2021 | TransMIL: Transformer Based Correlated Multiple Instance Learning for Whole Slide Image Classification;
Yun, S., Oh, S. J., Heo, B., Han, D., Choe, J., & Chun, S. | 2021 | Re-Labeling ImageNet: From Single to Multi-Labels, from Global to Localized Labels;
Eberle, O., Büttner, J., Kräutli, F., Müller, K.-R., Valleriani, M., & Montavon, G. | 2022 | Building and Interpreting Deep Similarity Models;
McInnes, L., Healy, J., Saul, N., & Großberger, L. | 2018 | UMAP: Uniform Manifold Approximation and Projection;
Karjauv, A. | 2026 | Rethinking Global Average Pooling: Your Classifier Is Secretly a Multi-Instance Learner;
{%- endcapture -%}

{% include reference.html ref=references %}

<script src="{{ '/assets/js/gap-mil.js' | relative_url }}" defer></script>
