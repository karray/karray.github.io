---
layout: post
title: "CNNs, ViTs, and the loudness of visual features"
description: "An interactive guide to feature length, direction, and context—and why a bright patch can mean different things in CNN and transformer maps."
math: true
tags: interpretability CNN ViT feature-geometry multiple-instance-learning
---

A dog stands on a lawn. We inspect an image classifier's spatial predictions and find “dog” written across the grass.

Has the model confused grass with a dog? Or does the feature at that location know about the dog standing nearby?

In the previous post, we saw how a classifier can hold more spatial evidence than its final answer reveals. Now we will look inside that evidence. Each spatial feature has a **length** and a **direction**. Its length acts a little like a volume knob; its direction determines which class scores it supports. The way information travels through the network helps explain both.

<!--more-->

<link rel="stylesheet" href="{{ '/assets/css/cnn-vit-geometry.css' | relative_url }}">
<script defer src="{{ '/assets/js/cnn-vit-geometry.js' | relative_url }}"></script>

This gives us a way to compare convolutional neural networks (**CNNs**) and vision transformers (**ViTs**) without assuming their maps should look alike. We will use simple geometric examples first, then return to measurements from the [GAP–MIL paper](https://arxiv.org/abs/2606.14555). All interactive figures are toy calculations, not outputs from a running neural network.

## A feature is an arrow

At one position in the final feature grid, the network stores a list of numbers: a **feature vector**. There might be hundreds of numbers, but start with just two. We can draw those two numbers as the coordinates of an arrow.

The arrow has two properties we can change separately:

- **Length**, also called its L2 norm: how far its tip is from the origin.
- **Direction**: the way it points, regardless of its length.

A classification head has its own weight vector for each class. It scores a feature using a **dot product**: how far the feature extends along the class's direction, multiplied by the length of the class weight vector. It can also add a constant called a bias.

Imagine a room of people describing a scene. Feature length is the speaker's volume. Direction describes what they are saying in the language the classifier reads. A louder voice has more influence, but volume alone tells you neither the message nor whether it is correct.

<figure class="cv-figure" id="cv-vector">
  <div class="cv-kicker">01 / Stretch the arrow. Then turn it.</div>
  <p>Here, “dog” reads the horizontal coordinate and “cat” reads the vertical coordinate. Both class weight vectors have length 1, and both biases are zero.</p>
  <div class="cv-columns">
    <svg id="cv-vector-svg" viewBox="0 0 360 285" role="img" aria-labelledby="cv-vector-title cv-vector-desc">
      <title id="cv-vector-title">A feature vector and its projection onto the dog direction</title>
      <desc id="cv-vector-desc">Length 2, angle 30 degrees. Dog score 1.73, cat score 1.00.</desc>
      <defs><marker id="cv-arrow-tip" markerWidth="7" markerHeight="7" refX="5.5" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7" fill="#14685f"/></marker></defs>
      <path d="M130 225 A50 50 0 0 1 230 225 M80 225 A100 100 0 0 1 280 225 M30 225 A150 150 0 0 1 330 225" fill="none" stroke="#dce4df" stroke-dasharray="3 4"/>
      <path d="M20 225 H340 M180 255 V60" stroke="#85958d" fill="none"/>
      <text x="302" y="250" class="cv-svg-label">dog →</text><text x="185" y="48" class="cv-svg-label">cat ↑</text><text x="164" y="243" class="cv-svg-label">0</text>
      <line id="cv-projection-drop" x1="266.60" y1="175" x2="266.60" y2="225" stroke="#ad581e" stroke-dasharray="4 4" stroke-width="2"/>
      <line id="cv-projection" x1="180" y1="225" x2="266.60" y2="225" stroke="#ad581e" stroke-width="5"/>
      <line id="cv-feature-arrow" x1="180" y1="225" x2="266.60" y2="175" stroke="#14685f" stroke-width="4" marker-end="url(#cv-arrow-tip)"/>
      <circle id="cv-feature-tip" cx="266.60" cy="175" r="4" fill="#14685f"/>
      <text x="20" y="278" class="cv-svg-label">Orange segment = dog score</text>
    </svg>
    <div>
      <label for="cv-length">Length <output id="cv-length-value" for="cv-length">2.0</output></label>
      <input id="cv-length" type="range" min="0" max="3" step="0.1" value="2" disabled>
      <label for="cv-angle">Direction <output id="cv-angle-value" for="cv-angle">30°</output></label>
      <input id="cv-angle" type="range" min="0" max="180" step="1" value="30" disabled>
      <div class="cv-readouts"><div>Dog score <strong id="cv-dog-score">1.73</strong></div><div>Cat score <strong id="cv-cat-score">1.00</strong></div><div>Dog alignment <strong id="cv-alignment">0.87</strong></div></div>
      <p id="cv-vector-result" class="cv-result" aria-live="polite">Dog wins. Stretching this arrow scales both scores; turning it changes their balance.</p>
    </div>
  </div>
  <noscript><p>Enable JavaScript to stretch and turn the arrow. The static example uses length 2 and direction 30°.</p></noscript>
  <figcaption>Try 90°: the dog score is zero at every length. Past 90°, a longer arrow makes that score more negative. At length zero there is no direction. This two-dimensional geometry illustrates an affine readout, not the arrangement of real dog and cat features.</figcaption>
</figure>

The dog score is the arrow's “shadow” on the dog axis. Lengthening an arrow increases the size of its projection, but the projection can be positive, zero, or negative. A loud feature can therefore lower a particular class score.

In the toy head, stretching a nonzero arrow leaves the winning class unchanged. In a real head, unequal biases can change that behavior. And even when a position's local winner stays the same, changing its length changes how strongly it participates in the pooled image prediction.

<details class="cv-details" markdown="1">
<summary>The geometry behind the volume knob</summary>

For nonzero feature $h$ and nonzero class vector $w_c$,

$$
s_c = w_c^\top h + b_c
    = \|h\|_2\,\|w_c\|_2\cos\theta_c + b_c.
$$

The cosine measures alignment: +1 means the same direction, 0 means perpendicular, and −1 means opposite directions. Classifier weight length and bias matter alongside feature length. This norm–angle decomposition also underlies normalized classifiers studied in [NormFace](https://arxiv.org/abs/1704.06369).

For deciding between two classes, the relevant quantity is their score difference:

$$
s_a-s_b=(w_a-w_b)^\top h+(b_a-b_b).
$$

A feature may project strongly onto both class vectors and still barely distinguish them. “Loudness” is shorthand for scaling a contribution in a fixed feature space and head, not a probability, certainty score, or measure of information content.

</details>

## GAP gives every position a seat, but not the same voice

Global average pooling gives each position the same numerical weight, $1/N$, where $N$ is the number of positions. But it averages vectors with different lengths and directions. Equal pooling weights do not make their contributions equally large.

Write a nonzero feature as its length times its unit direction: $h_i=r_i u_i$. Then the average is

$$
\bar h=\frac{1}{N}\sum_i r_i u_i.
$$

So the lengths act as multipliers on the directional evidence. Several long arrows pointing together can dominate shorter arrows; long arrows pointing in opposing directions can cancel. Length alone cannot tell us which case we have.

This connects back to LaFAM. It uses a channel average to expose feature activity without choosing a class. For nonnegative activations, that average is proportional to the L1 norm—the sum of magnitudes. L2 length is related, but is a different measure. Neither includes the classifier's directional preferences. [LaFAM paper](https://arxiv.org/abs/2407.06059).

A magnitude map can be useful when an object's features are stronger than those of its surroundings. A class map answers a different question: how do those features project onto a particular class vector? It can look different even before we change architectures.

## CNNs and ViTs give the observers different ways to talk

A typical convolution combines nearby features using learned filters. Stacking layers expands the region that can influence a position. Think of observers passing notes to their neighbors: after several rounds, a report can contain news from far away.

That region is called a **receptive field**. Its theoretical extent describes where information could come from; its effective extent describes where influence is concentrated. Those can differ considerably. A late CNN feature can depend on much more than the small image square beneath it. [Luo et al., *Understanding the Effective Receptive Field*](https://arxiv.org/abs/1701.04128).

A standard ViT starts with image patches represented as vectors, called **tokens**. Self-attention allows a token to combine transformed information from other tokens using weights that depend on the input. The observers now have a shared conversation: a report about the grass can receive information from the dog without waiting for a chain of neighboring reports. [Dosovitskiy et al., *An Image Is Worth 16×16 Words*](https://arxiv.org/abs/2010.11929).

This is a difference in communication, not a division between models that use context and models that cannot. Attention can favor nearby positions, and some transformer architectures restrict attention to windows. CNNs can gather broad context through depth and their choice of operations. [Raghu et al.](https://arxiv.org/abs/2108.08810) found early global information aggregation in the ViTs they studied, while also finding that spatial information could be preserved.

The consequence for interpretation is subtle: **a feature's address tells us where it is stored, not everything it describes**. “Dog” at a grass position might mean “this report includes dog evidence.” It need not mean “these grass pixels look like a dog.”

## Two ways to change a background vote

Suppose some positions favor a competing class while a small region favors the dog. There are at least two ways the representation could make the average favor the dog.

One is to **turn down the competing features**. Their lengths decrease, so they contribute less. The direction can stay exactly the same: a quiet competing vote is still a competing vote.

Another is to **change their content using context**. Information from the object reaches other positions, changing their vectors and their alignment with the dog classifier. Those positions can now add to the dog score rather than simply becoming quieter.

<figure class="cv-figure" id="cv-field">
  <div class="cv-kicker">02 / Quieter evidence or different evidence?</div>
  <p>Four outlined positions represent the dog region, with vector (3, 0). The other twelve start at (0, 1.5), favoring a competitor. Choose a change and increase its strength.</p>
  <div class="cv-buttons" role="group" aria-label="Toy mechanism">
    <button type="button" id="cv-mode-quiet" aria-pressed="true" disabled>Lower background volume</button>
    <button type="button" id="cv-mode-context" aria-pressed="false" disabled>Share object context</button>
  </div>
  <label for="cv-strength">Change strength <output id="cv-strength-value" for="cv-strength">60%</output></label>
  <input id="cv-strength" type="range" min="0" max="100" step="1" value="60" disabled>
  <div class="cv-maps">
    <div><h3>Length</h3><div class="cv-map" id="cv-map-norm" role="img" aria-label="Feature lengths: dog positions 3; background 0.74"><span aria-hidden="true" style="background:rgb(185,209,202);color:#22372c">0.74</span><span aria-hidden="true" style="background:rgb(185,209,202);color:#22372c">0.74</span><span aria-hidden="true" style="background:rgb(185,209,202);color:#22372c">0.74</span><span aria-hidden="true" style="background:rgb(185,209,202);color:#22372c">0.74</span><span aria-hidden="true" style="background:rgb(185,209,202);color:#22372c">0.74</span><span class="cv-object" aria-hidden="true" style="background:rgb(20,104,95);color:#ffffff">3.00</span><span class="cv-object" aria-hidden="true" style="background:rgb(20,104,95);color:#ffffff">3.00</span><span aria-hidden="true" style="background:rgb(185,209,202);color:#22372c">0.74</span><span aria-hidden="true" style="background:rgb(185,209,202);color:#22372c">0.74</span><span class="cv-object" aria-hidden="true" style="background:rgb(20,104,95);color:#ffffff">3.00</span><span class="cv-object" aria-hidden="true" style="background:rgb(20,104,95);color:#ffffff">3.00</span><span aria-hidden="true" style="background:rgb(185,209,202);color:#22372c">0.74</span><span aria-hidden="true" style="background:rgb(185,209,202);color:#22372c">0.74</span><span aria-hidden="true" style="background:rgb(185,209,202);color:#22372c">0.74</span><span aria-hidden="true" style="background:rgb(185,209,202);color:#22372c">0.74</span><span aria-hidden="true" style="background:rgb(185,209,202);color:#22372c">0.74</span></div><div class="cv-scale"><span>0</span><span>3</span></div></div>
    <div><h3>Dog alignment</h3><div class="cv-map" id="cv-map-angle" role="img" aria-label="Cosine with dog direction: dog positions 1; background 0"><span aria-hidden="true" style="background:rgb(238,243,237);color:#22372c">0.00</span><span aria-hidden="true" style="background:rgb(238,243,237);color:#22372c">0.00</span><span aria-hidden="true" style="background:rgb(238,243,237);color:#22372c">0.00</span><span aria-hidden="true" style="background:rgb(238,243,237);color:#22372c">0.00</span><span aria-hidden="true" style="background:rgb(238,243,237);color:#22372c">0.00</span><span class="cv-object" aria-hidden="true" style="background:rgb(113,81,142);color:#ffffff">1.00</span><span class="cv-object" aria-hidden="true" style="background:rgb(113,81,142);color:#ffffff">1.00</span><span aria-hidden="true" style="background:rgb(238,243,237);color:#22372c">0.00</span><span aria-hidden="true" style="background:rgb(238,243,237);color:#22372c">0.00</span><span class="cv-object" aria-hidden="true" style="background:rgb(113,81,142);color:#ffffff">1.00</span><span class="cv-object" aria-hidden="true" style="background:rgb(113,81,142);color:#ffffff">1.00</span><span aria-hidden="true" style="background:rgb(238,243,237);color:#22372c">0.00</span><span aria-hidden="true" style="background:rgb(238,243,237);color:#22372c">0.00</span><span aria-hidden="true" style="background:rgb(238,243,237);color:#22372c">0.00</span><span aria-hidden="true" style="background:rgb(238,243,237);color:#22372c">0.00</span><span aria-hidden="true" style="background:rgb(238,243,237);color:#22372c">0.00</span></div><div class="cv-scale cv-scale-purple"><span>0</span><span>1</span></div></div>
    <div><h3>Dog score</h3><div class="cv-map" id="cv-map-score" role="img" aria-label="Dog scores: dog positions 3; background 0"><span aria-hidden="true" style="background:rgb(238,243,237);color:#22372c">0.00</span><span aria-hidden="true" style="background:rgb(238,243,237);color:#22372c">0.00</span><span aria-hidden="true" style="background:rgb(238,243,237);color:#22372c">0.00</span><span aria-hidden="true" style="background:rgb(238,243,237);color:#22372c">0.00</span><span aria-hidden="true" style="background:rgb(238,243,237);color:#22372c">0.00</span><span class="cv-object" aria-hidden="true" style="background:rgb(20,104,95);color:#ffffff">3.00</span><span class="cv-object" aria-hidden="true" style="background:rgb(20,104,95);color:#ffffff">3.00</span><span aria-hidden="true" style="background:rgb(238,243,237);color:#22372c">0.00</span><span aria-hidden="true" style="background:rgb(238,243,237);color:#22372c">0.00</span><span class="cv-object" aria-hidden="true" style="background:rgb(20,104,95);color:#ffffff">3.00</span><span class="cv-object" aria-hidden="true" style="background:rgb(20,104,95);color:#ffffff">3.00</span><span aria-hidden="true" style="background:rgb(238,243,237);color:#22372c">0.00</span><span aria-hidden="true" style="background:rgb(238,243,237);color:#22372c">0.00</span><span aria-hidden="true" style="background:rgb(238,243,237);color:#22372c">0.00</span><span aria-hidden="true" style="background:rgb(238,243,237);color:#22372c">0.00</span><span aria-hidden="true" style="background:rgb(238,243,237);color:#22372c">0.00</span></div><div class="cv-scale"><span>0</span><span>3</span></div></div>
  </div>
  <p class="cv-map-key">Outlined cells are the fixed object region. Numbers are per-position values; color scales stay fixed.</p>
  <p id="cv-field-detail" class="cv-result" aria-live="polite">Background length falls to 0.74, but dog alignment stays 0.00. Its dog score remains 0.00.</p>
  <div class="cv-readouts"><div>Mean dog score <strong id="cv-field-dog">0.75</strong></div><div>Mean competitor score <strong id="cv-field-other">0.55</strong></div><div>Image winner <strong id="cv-field-winner">dog</strong></div></div>
  <noscript><p>Enable JavaScript to compare the changes. This static view shows background volume reduced at 60% strength.</p></noscript>
  <figcaption>Two constructed feature changes, not CNN and ViT simulations. Both can change the image prediction. In the context example, a background feature initially becomes shorter even as its dog score rises: length and alignment need to be read together.</figcaption>
</figure>

<details class="cv-details" markdown="1">
<summary>What the slider computes</summary>

With strength $t$ between 0 and 1, the first mode multiplies each background vector by $1-0.85t$. The second uses $\alpha=0.8t$ to mix the starting background vector with the object vector:

$$
h_{\rm background}=(1-\alpha)(0,1.5)+\alpha(3,0).
$$

The central four vectors stay fixed. The head reads the first coordinate for dog and the second for its competitor. All biases are zero. The maps show L2 length, cosine with the dog direction, and their product. This simple mixture illustrates a possible geometric effect of context; an actual attention block also has learned projections, multiple heads, residual paths, and other operations.

</details>

Reduced competing contributions and wider sharing of object evidence are useful hypotheses when comparing CNN and transformer maps. Neither is reserved for one architecture, and neither follows inevitably from image-level supervision. A network may combine them, preserve uncertainty, or learn a different solution.

They also make different predictions about what we should see. In the first case, background can remain semantically distinct but have little influence on the mean. In the second, an object class can win far beyond its visible outline. A diffuse class map could coexist with a correct image prediction and preserved information about spatial layout.

## Remove the dog, then look at the grass again

To investigate that second possibility, the GAP–MIL paper hides foreground pixels and runs the modified image through the model again. Its background-activation metric asks whether the target class wins at least one grid cell outside the annotated foreground.

The rate drops in all five reported ImageNet models, with larger drops for Swin and MaxViT. The values below come from [Table 1](https://arxiv.org/html/2606.14555v1#S5.T1).

<figure class="cv-figure" id="cv-evidence">
  <div class="cv-kicker">03 / Background readings depend on the visible object</div>
  <p><strong>Images where the target wins somewhere in the background</strong></p>
  <p class="cv-evidence-legend"><span class="cv-original-key"></span> Original image <span class="cv-masked-key"></span> Foreground hidden</p>
  <div class="cv-evidence-row"><div class="cv-evidence-label"><span>EfficientNet <small>CNN</small></span><strong>55.3% → 24.6%</strong></div><div class="cv-evidence-track" aria-hidden="true"><span style="width:55.3%"></span><span style="width:24.6%"></span></div></div>
  <div class="cv-evidence-row"><div class="cv-evidence-label"><span>ResNet-50 <small>CNN</small></span><strong>34.5% → 28.7%</strong></div><div class="cv-evidence-track" aria-hidden="true"><span style="width:34.5%"></span><span style="width:28.7%"></span></div></div>
  <div class="cv-evidence-row"><div class="cv-evidence-label"><span>ConvNeXt <small>CNN</small></span><strong>55.2% → 34.7%</strong></div><div class="cv-evidence-track" aria-hidden="true"><span style="width:55.2%"></span><span style="width:34.7%"></span></div></div>
  <div class="cv-evidence-row"><div class="cv-evidence-label"><span>Swin <small>windowed attention</small></span><strong>89.2% → 35.8%</strong></div><div class="cv-evidence-track" aria-hidden="true"><span style="width:89.2%"></span><span style="width:35.8%"></span></div></div>
  <div class="cv-evidence-row"><div class="cv-evidence-label"><span>MaxViT <small>convolution + attention</small></span><strong>90.3% → 37.9%</strong></div><div class="cv-evidence-track" aria-hidden="true"><span style="width:90.3%"></span><span style="width:37.9%"></span></div></div>
  <div class="cv-axis"><span>0%</span><span>50%</span><span>100%</span></div>
  <figcaption>Reported ImageNet validation measurements. The percentages count images, not background cells, and are not feature norms. Foreground comes from bounding boxes projected to each model's grid. Swin uses windowed attention; MaxViT combines convolution and attention. These are not five otherwise identical models.</figcaption>
</figure>

The result is **consistent with object information contributing to background features**. It does not prove that every background prediction came from the object: masking changes the input distribution, boxes are imperfect, and context itself can predict the class.

There is a useful distinction between two experiments. Hiding input pixels and recomputing features tests a response to changed visual evidence. Deleting a final feature vector tests a change to the aggregation. By that late stage, information from the deleted region may already exist elsewhere.

To go further, we could keep a grass patch unchanged while varying the surrounding object, then track its feature across layers. Separately, we could keep the dog and replace its background. Together, these comparisons would help distinguish useful context exchange from reliance on a particular setting. A single heatmap cannot settle that question.

## Does LayerNorm remove the volume knob?

Transformers commonly use **LayerNorm**, so it is tempting to say their token lengths must all be the same.

LayerNorm first subtracts a vector's channel mean and rescales by its channel standard deviation. At that intermediate stage, the length is approximately fixed for a given number of channels. But the usual learned version then applies a separate gain and bias to each channel. These can make the output lengths different again. [Ba et al., *Layer Normalization*](https://arxiv.org/abs/1607.06450).

Think of standardizing the volume of several recordings, then passing them through the same equalizer. Boosting bass affects a bass-heavy recording more than a flute solo. The equalizer settings are shared; the resulting loudness still depends on the recording.

<figure class="cv-figure" id="cv-layernorm">
  <div class="cv-kicker">04 / Equal lengths before the equalizer</div>
  <p>Two three-channel vectors have already been standardized. Both have zero mean, variance 1, and length √3 ≈ 1.73. Apply the same gain to their first channel.</p>
  <div class="cv-token-pair"><div><strong>Token A</strong><code>(1.225, −1.225, 0)</code></div><div><strong>Token B</strong><code>(0.707, 0.707, −1.414)</code></div></div>
  <label for="cv-gain">First-channel gain <output id="cv-gain-value" for="cv-gain">3.0×</output></label>
  <input id="cv-gain" type="range" min="1" max="4" step="0.1" value="3" disabled>
  <div class="cv-norm-row"><span>Output length A</span><strong id="cv-norm-a">3.87</strong><div class="cv-track"><div id="cv-norm-a-bar" style="width:73.075%"></div></div></div>
  <div class="cv-norm-row cv-purple"><span>Output length B</span><strong id="cv-norm-b">2.65</strong><div class="cv-track"><div id="cv-norm-b-bar" style="width:49.92%"></div></div></div>
  <p id="cv-ln-result" class="cv-result" aria-live="polite">The same gain produces different lengths because the vectors put different amounts into the first channel.</p>
  <noscript><p>Enable JavaScript to change the gain. At gain 1 both lengths are 1.73; at gain 3 they become 3.87 and 2.65.</p></noscript>
  <figcaption>An exact toy calculation using A = (√1.5, −√1.5, 0) and B = (√0.5, √0.5, −√2); displayed coordinates are rounded. Other gains are 1 and all biases are 0. No second standardization is applied after the gain.</figcaption>
</figure>

<details class="cv-details" markdown="1">
<summary>Where to measure the norm</summary>

The usual operation is

$$
\hat h_j=\frac{h_j-\mu}{\sqrt{\sigma^2+\epsilon}},\qquad
\operatorname{LN}(h)_j=\gamma_j\hat h_j+\beta_j.
$$

Ignoring the small stabilizer $\epsilon$, a nonconstant vector standardized across $d$ channels has squared length $d$. The learned channel gains $\gamma_j$ and biases $\beta_j$ need not preserve that length or direction. Before LayerNorm, after standardization, and after its learned transformation are three different measurement points.

Literal L2 normalization, $h/\|h\|_2$, does fix the length of nonzero vectors to 1 at that point. It is a different operation.

</details>

There is also published evidence that a transformer can have particularly *long* background features. [*Vision Transformers Need Registers*](https://arxiv.org/abs/2309.16588) identifies high-norm background tokens used for internal computation in the models it studies. The work adds extra tokens to provide space for that computation.

That result is a good reason to resist equating length with objectness. A large vector can carry useful global information while being a poor description of the local pixels. It does not follow that all transformer background features serve this role, or that this mechanism explains the occlusion results above.

## Read the length, direction, and address together

For a spatial feature, we now have three separate questions:

| Question | What to inspect |
|---|---|
| How strongly can it contribute? | Its length, the classifier weights, and the actual score or margin. |
| Which classes does it favor? | Its alignment with the head, including competing class directions. |
| Where did its information come from? | The network's communication paths and responses to input changes. |

For a CNN, a magnitude map can be a useful first look, as LaFAM showed. It still needs a class-specific reading when we ask what evidence the classifier uses. For a ViT, context exchange makes it especially important to distinguish a token's location from the origin of its information. Normalization and extra tokens further affect what its length means. These are reasons to inspect the model's actual computation rather than interpret all maps by the same visual rule.

The pooling path matters too. The previous post's exact mean-score identity applies to mean-pooled spatial features followed by an affine head. A ViT that classifies from a special class token does not automatically have that decomposition. Applying its head to patch tokens can be a diagnostic, but we must check what that readout represents.

One final practical point: raw norms from different networks are not in a common unit. Multiplying all features by ten and dividing the head weights by ten preserves every logit while changing every feature length. More general coordinate changes can alter angles too. Compare geometry at a specified layer and normalization point, with its head attached.

Back on the lawn, “dog” at a grass position has become a more interesting observation. The feature might be quiet or loud, locally distinctive or rich in shared context. Its position alone does not answer those questions. Length tells us about scale, direction tells us about the classifier's reading, and changing the image helps us trace the information that produced both.

<!-- Editorial: follows _drafts/your-classifier-is-already-a-multiple-instance-learner.md,
which follows the LaFAM draft on branch `lafam`. Add series links when publication
URLs are fixed. The follow-up audit informed scope and caveats only; this post
reports no unpublished audit measurements. -->
