---
layout: post
title: "Following relevance through a network with no labels"
description: "An interactive journey through label-free LRP, from promising object maps to a stubborn patch grid."
permalink: /drafts/lrp-without-labels/
tags: [LRP, Explainability, Self-Supervised Learning, Vision Transformers]
---

A vision model can recognize a surprising amount about a photograph without ever having been taught the word *cat*. But ask where that knowledge lives in the image, and things get awkward. There may be no cat score to explain. Just a long list of numbers: an embedding.

Layer-wise relevance propagation, or **LRP**, gives us a way to follow amounts assigned to those numbers back toward the pixels. It sounds like a plumbing problem. Choose where to pour; follow the flow. In practice, some pipes split, some subtract, and some lead to places that are not pixels at all.

This is the story of making that work across modern vision models—and of one model whose patch grid refused to go away.

<!--more-->

<link rel="stylesheet" href="{{ '/assets/css/posts/lrp-without-labels.css' | relative_url }}">
<div id="lrp-article" data-source="{{ '/assets/downloads/lrp-without-labels/data.json' | relative_url }}"></div>

## First, something to look at

Start with the dog and cat. Switch between DINOv1, DINOv2, DINOv3 and MAE. These are recorded explanations, not a simulation running in your browser. None starts from a class label or a selected object. Each starts with **one unit, divided equally across every spatial feature coordinate** at the chosen output.

<div class="lrp-widget" id="lrp-gallery">
{% include figure-gallery.html id="lrp-model-maps" data=site.data.galleries.lrp_models %}
<p class="lrp-readout" data-role="scale"></p>
<p class="lrp-readout" data-role="ledger" aria-live="polite">Loading recorded amounts…</p>
<p class="lrp-note">Signed colors: blue negative, white zero, red positive. Positive-only colors: black zero to yellow/white at the maximum. Each panel scales independently. Positive-only display hides negative channels; it does not change propagation. Signed pixel totals include all RGB channels, before cancellation.</p>
</div>

Recognizable objects appear in several maps. Some maps favor background or isolated spots. MAE often draws a conspicuous grid. Those are observations to investigate, rather than a beauty contest with an obvious winner. A model might really use a background cue. A high-norm token—an unusually large feature vector—might really dominate its representation. An explanation should let us discover that.

The explorer includes supervised models and CLIP as comparisons. CLIP learns from image–text pairs; it is not an image-only self-supervised method. Architecture, training data, objective and propagation rules differ across these checkpoints. This gallery does not isolate the effect of any one of them. [Recorded configurations and amounts]({{ '/assets/downloads/lrp-without-labels/backbone-heatmaps.json' | relative_url }}).

## You do not need a cat score

LRP's original formulation focused on explaining classifier decisions: begin with a prediction and redistribute it backward through the network. [Bach and colleagues, 2015](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140).

But the backward calculation can begin inside the network. A layer contains numbers; we assign relevance to some or all of them; the rules redistribute those amounts toward the input. There need not be a class, prompt, reference image or similarity score. The experiments here use that flexibility explicitly.

An **embedding** is a vector of learned measurements. A spatial encoder gives us one such vector at each location. In a vision transformer, the image starts as patches, and each patch becomes a vector called a **token**. Later layers exchange information between tokens, so a token's original location does not mean its contents describe only that patch.

For our MAE checkpoint, the output contains 196 spatial tokens with 768 coordinates each. Three possible starting choices are:

- **One unit spread uniformly:** give each of the 150,528 coordinates the same amount. This asks how the chosen rules route an equal allocation across the representation.
- **The full embedding mean:** give each coordinate its value divided by 150,528. Positive and negative values both participate; their total is the embedding mean.
- **The mean of positive values:** use only positive coordinates, dividing by the number of strictly positive values. The total is their mean, not necessarily one.

These are different questions. Equal allocation does not mean equal influence after propagation. Averaging coordinates does not turn an embedding into a named concept. And selecting only positive values changes the starting explanation; it does not repair negative relevance after the fact.

You can also begin with a particular channel or an intermediate layer. That makes the starting point more specific, but calling a channel “the cat concept” would require additional evidence.

<details markdown="1">
<summary>What does “one unit” actually mean?</summary>

It is a chosen total, not a physical measurement or a hidden probability. If the initial tensor has $N$ entries, uniform allocation means $R_i=1/N$ and $\sum_i R_i=1$. For the raw mean, $R_i=z_i/N$. For the positive-only mean, $R_i=\max(z_i,0)/\#\{z_j>0\}$, with zero allocation if that count is zero.

The MAE experiments initialize all spatial tokens and exclude the special CLS token. CLS can still receive relevance during propagation and needs separate accounting. We never select object-shaped patch subsets for the examples in this post.

The water analogy works for positive amounts and splits. With signed amounts, think of a ledger with credits and debits. A small net total can conceal large opposing flows.

</details>

## Every kind of junction needs a rule

A network's ordinary backward pass computes derivatives: how a small change in one number affects another. LRP's backward pass redistributes an amount. Those operations can share implementation machinery, but they do not answer the same question.

Consider a residual connection, which adds a transformed signal to a shortcut. If the incoming relevance is handed in full to both branches, we have copied it. A real split must decide how much each branch receives. Equal halves? Shares based on contribution magnitudes? Signed proportions? Each is an assumption about attribution.

This matters even when the picture looks fine. In our matched ResNet controls, splitting equally at 16 joins gives the same map shape as duplicating at every join, after dividing the duplicate map by **65,536**. Independently scaling the colors hides the difference. A familiar silhouette cannot tell us whether the bookkeeping is sensible. [Residual controls]({{ '/assets/downloads/lrp-without-labels/backbone-validation.json' | relative_url }}).

For ConvNeXt, getting the machinery right meant explicitly handling timm's GELU activation, residual additions and learned layer scales, as well as choosing rules for mixing layers and the input projection. The GELU omission was real, but fixing it alone made only a modest difference in the measured selectivity. There was no single magic missing hook.

Transformers add more junctions: attention mixes tokens using weights computed from those same tokens; normalization centers and rescales features; positional embeddings and special tokens introduce context beyond image pixels. Work such as [AttnLRP](https://proceedings.mlr.press/v235/achtibat24a.html) develops rules for transformer operations and enables analysis of internal representations. Our experiments also explore alternatives; the gallery should not be read as a reproduction of one published rule set across every architecture.

<details markdown="1">
<summary>The practical work behind “supporting a model”</summary>

| Operation | What needed an explicit decision or check |
|---|---|
| Convolution / linear projection | How activations and weights divide the incoming amount; how bias is accounted for |
| GELU / other nonlinearities | An explicit matching rule, including timm-specific types |
| Residual addition | Split the amount rather than accidentally copying it |
| LayerNorm | Centering, learned scale and bias, and how to handle the input-dependent variance |
| Attention | Whether and how to attribute the attention weights as well as the values being mixed |
| Position / prefix tokens | Which amounts go to image content and which go to context |
| Reshape / window operations | Preserve exact spatial alignment and cover the actual implementation |

A “composite” is simply this collection of rules. Forward-equivalence checks ensure that installing it leaves predictions unchanged; cleanup checks ensure the model is restored afterward. Neither establishes that its attribution assumptions are correct.

The ConvNeXt classifier study improved a development-set top-K overlap measure from 0.518 for the original ResNet reference to 0.702. The masks were predictions from another model, rule choices used the same examples, and only about 0.65 of a unit reached pixels. These are useful debugging results, not a held-out faithfulness benchmark. [Detailed source report]({{ '/assets/downloads/lrp-without-labels/convnext-findings.md' | relative_url }}).

</details>

## What happens in between is not just a blurrier version of the ends

Every map so far has been an input-level pixel map or a starting embedding. None of them show what relevance looks like partway through the network. For a stack of convolutions, that turns out to matter: one architecture carries a real, structural checkerboard through its intermediate stages, and another does not. To check that this is about the architecture and not about how the relevance got started, everything below uses the same uniform, label-free allocation as the rest of this post—on a supervised **and** a self-supervised checkpoint of each architecture.

Choose any of the five images, then switch between ResNet50 and ConvNeXt-B, each supervised and each trained with DINO, at each stage output from right after the stem to the last stage before pooling.

<div class="lrp-widget" id="lrp-intermediate">
{% include figure-gallery.html id="lrp-intermediate-maps" data=site.data.galleries.lrp_intermediate %}
<p data-role="scale" class="lrp-readout"></p>
<p class="lrp-readout" data-role="stage" aria-live="polite"></p>
<p class="lrp-readout" data-role="amount"></p>
<p class="lrp-note">Positive channel mean, clipped and scaled independently at each depth: black means small or nonpositive relevance, not necessarily zero. Depth 4 is where the uniform initial relevance itself lives—every map there is flat by construction; the interesting question is what a network does to it on the way back to depth 0.</p>
</div>

A **phase ratio** compares the strongest and weakest of the four spatial phases a stride-2 layer produces (even/even, even/odd, odd/even, odd/odd); 1.0 means no imbalance. Both ResNet checkpoints start close to **9** right after the stem and stay above **4.7** through the middle stages—supervised and self-supervised alike. Both ConvNeXt checkpoints never leave **1.0–1.2**. Training objective does not decide which architecture shows this; the architecture does. [Recorded stage statistics]({{ '/assets/downloads/lrp-without-labels/intermediate-stages.npz' | relative_url }}).

### Two branches, opposite phase

ResNet's imbalance has a mechanism, not just a measurement. Every stage transition (for example `layer4.0`) adds a **main branch**, a 3 × 3 stride-2 convolution, to a **shortcut**, a 1 × 1 stride-2 convolution. Isolate each branch's relevance at the transition it feeds, again on any image and for both the supervised and the DINO checkpoint, and compare an unsplit residual against one split by activation magnitude:

<div class="lrp-widget" id="lrp-checkerboard">
{% include figure-gallery.html id="lrp-checkerboard-maps" data=site.data.galleries.lrp_checkerboard %}
<p data-role="scale" class="lrp-readout"></p>
<p class="lrp-readout" data-role="result" aria-live="polite"></p>
<p class="lrp-note">Absolute channel-summed relevance; independent full-range scales.</p>
</div>

The main branch's 3 × 3 windows are centered on even positions, so an odd/odd position sits inside four overlapping windows against one for even/even. The shortcut's 1 × 1 window cannot overlap at all—it returns relevance **only** at even/even positions; the other three quarters of positions get exactly zero from it, a genuine disconnection rather than noise, not something a different rule could route around. Both patterns are structural, not content-dependent: they show up identically under unit activations and uniform incoming relevance, with no photograph involved. [Structure-only controls for both checkpoints]({{ '/assets/downloads/lrp-without-labels/intermediate-findings.md' | relative_url }}).

The visible checkerboard is their weighted sum, and here is a wrinkle worth being honest about: this post's own production rule for splitting a ResNet residual—**equal**, a fixed 50/50—turns out not to change that sum at all. Dividing both branches by the same constant leaves their *relative* weighting exactly where it was; it is mathematically indistinguishable from not splitting. Only a rule that treats the branches asymmetrically moves anything. Switching to a **magnitude**-weighted split—each branch's share proportional to its own activation size at that position—lowers the even/even share from **0.571** to **0.429** on the supervised checkpoint and from **0.560** to **0.374** on the DINO checkpoint. Real, and the same direction for both training regimes, but a rule this post does not actually use anywhere else. [Full numbers and reproduction]({{ '/assets/downloads/lrp-without-labels/intermediate-findings.md' | relative_url }}).

ConvNeXt has no equivalent mechanism: every strided layer there has kernel size equal to stride, so each position is covered exactly once.

<details markdown="1">
<summary>Is this worth fixing?</summary>

Not yet decided, and not obviously wrong as it stands: the shortcut genuinely never reads three quarters of the positions, and the rule genuinely gives shared positions more. The final pixel maps stay clean regardless (patch-edge ratio close to 1.0); this only matters when an intermediate map is read directly. Options under consideration, from least to most intrusive: report the two branches separately; normalize the displayed map by the phase pattern measured under unit activations, which changes the display but not the relevance; or replace the shortcut with average-pooling followed by a 1 × 1 convolution (the ResNet-D layout)—an untested surrogate that would need its own prediction checks. Hierarchical vision transformers with their own strided downsampling, such as Swin, have not been checked for an analogous effect.

This is the same kind of artifact [Odena and colleagues](https://distill.pub/2016/deconv-checkerboard/) describe for transposed convolutions—the operation a strided convolution's backward pass performs—though AttnLRP's paper uses the word "checkerboard" for something unrelated: the token-sized blocks of an attention-rollout map, not this uneven-overlap pattern.

</details>

## The last step can stamp a pattern onto every patch

Going backward from one patch token to a 16 × 16 patch means distributing its amount among 256 pixels and their color channels. We do not simply enlarge a square on a screen. The input projection and its attribution rule decide how to fill it.

One rule, **WSquare**, distributes according to squared projection weights. Once the incoming relevance is fixed, this local distribution does not depend on the image values. It can repeatedly stamp a learned template inside patches. **ZPlus** also uses the input activations, so its distribution can change with the local image.

Compare the two input-projection rules side by side, then switch models using the image dock below. The incoming patch relevance is identical in each paired experiment; only the final redistribution changes.

<div class="lrp-widget" id="lrp-stem">
{% include figure-gallery.html id="lrp-projection-maps" data=site.data.galleries.lrp_stems %}
<p data-role="scale" class="lrp-readout"></p>
<p data-role="result" aria-live="polite"></p>
</div>

On this montage, the fraction of within-patch map energy captured by one repeated template drops from **99.6% to 82.8%** for DINOv2 and **99.3% to 84.0%** for DINOv3. That explains a visible change in texture. It does not prove that ZPlus identifies the right pixels, and it does not eliminate all patch structure. [Paired measurements]({{ '/assets/downloads/lrp-without-labels/backbone-validation.json' | relative_url }}).

There is also unfinished business upstream: different DINOv2 output-channel allocations produced strongly correlated pixel maps. A detailed-looking map can lose distinctions that were present in the representation. Better texture and better specificity are separate achievements.

## MAE: the grid has an accomplice

[Masked autoencoders](https://arxiv.org/abs/2111.06377) learn by reconstructing missing image patches. Here we use the trained encoder on the complete image, with no random masking and no decoder. [DINO](https://arxiv.org/abs/2104.14294), by contrast, uses self-distillation; its authors reported emergent spatial structure in self-supervised ViTs. Different learning objectives make different representations plausible. They do not, by themselves, explain our attribution results.

MAE's projection weights provide a more concrete clue. A patch's outermost rows and columns occupy **23.44%** of its area, but contain **54.30%** of MAE's squared projection weights. The tested DINOv1 and supervised ViT checkpoints are near the area fraction: **22.55%** and **23.15%**.

This boundary preference also appears outside LRP. Native gradient energy concentrates at patch edges; matched perturbations to boundary pixels change MAE's embeddings more than equally sized, equal-energy interior perturbations in the tested pairs. The grid has a model-dependent component. We have not established which part of training caused it. [Boundary investigation and limits]({{ '/assets/downloads/lrp-without-labels/mae-findings.md' | relative_url }}).

But where does the grid enter the explanation? Follow an actual backward trace. Start at the final embedding, then move toward the pixels. Compare the last two stages: the pixel map and those same pixel amounts added back together within each original patch.

<div class="lrp-widget" id="lrp-trace">
<p data-role="stage" aria-live="polite"></p>
{% include figure-gallery.html id="lrp-backward-maps" data=site.data.galleries.lrp_trace %}
<p data-role="scale" class="lrp-readout"></p>
<p class="lrp-readout" data-role="amount"></p>
<p class="lrp-note">Blue is negative, red positive. Each stage has its own symmetric full-range color scale. Coarse panels are actual 14 × 14 channel sums enlarged without smoothing; they are not pixel explanations or selected-patch masks.</p>
<div class="lrp-bias"><strong>Why is the full mean negative?</strong><p data-role="bias" aria-live="polite"></p></div>
</div>

Across these six cases, the incoming patch amounts and the pixel amounts summed back within their patches correlate **0.994–0.9999**. Much of the conspicuous fine grid arises *inside* the patches. Coarse structure can survive while the pixel redistribution looks unhelpful. That still leaves the coarse ranking itself to validate. [Recorded stage diagnosis]({{ '/assets/downloads/lrp-without-labels/mae-diagnosis.json' | relative_url }}).

### A negative mean is not automatically “evidence against the object”

MAE's final normalization includes a learned bias. Averaged over coordinates, that bias is **−0.03298339**, the same for every image. On the dog/cat input, the image-dependent term is about **+0.02331165**. Add the bias and the full embedding mean is **−0.00967174**.

The negative total does not mean the dog and cat are negative concepts. It partly reflects where the model places the origin of its feature coordinates. The exact output decomposition above exposes that offset. We keep the initial relevance unchanged and account for parameter contributions separately; we do not flip the finished heatmap to make it look nicer.

For positive-only means, the set of participating coordinates changes with the image. Their bias contribution need not be a constant spatial offset. Switch the allocation in the trace to see the measured decomposition change. [Exact normalization decomposition]({{ '/assets/downloads/lrp-without-labels/mae-norm.json' | relative_url }}).

<details markdown="1">
<summary>Why not rescale embeddings to −1…+1 first?</summary>

That defines another allocation. It may be useful if its meaning fits the question, but it is not neutral housekeeping. In our three MAE examples, a global min–max transform changed the fraction of positive coordinates from roughly 46–47% to **0.362%, 99.351% and 99.759%**. One extreme coordinate can move the zero point for almost everything else.

Similarly, stabilizing a denominator can change the attribution rather than merely compute it more accurately. Increasing the final normalization stabilizer improved MAE's numerical robustness, yet made different channel explanations more alike and weakened one patch-replacement test. Stability is necessary; it is not sufficient. [Normalization experiments]({{ '/assets/downloads/lrp-without-labels/mae-findings.md' | relative_url }}).

</details>

## Could we just take the shortcut?

A transformer has a residual stream carrying information around its attention and feature-mixing blocks. It was tempting to route relevance through that stream and bypass the difficult operations.

We tried seven backward configurations on three images and both embedding means: 42 cases. Identity-only routing, skipping attention, skipping feature mixing, equal splits and other alternatives all retained the pixel grid. The forward network was identical; these were changes to the attribution assumptions, not models with layers removed.

There is a reason the shortcut cannot settle the question. Preserving an amount on one branch tells us nothing about whether assigning zero to the other branch is justified. And none of those routes avoids the final step from patch features into pixels.

Other tempting improvements failed for different reasons. Selecting object-like patch groups largely reproduced the selection itself. Averaging explanations from shifted images reduced grid visibility, but explained multiple altered inputs and did not establish better faithfulness. Changing the pixel rule to respect physical RGB bounds improved invariance to coordinate conventions, yet kept the grid. These are informative failures, not grounds for declaring every future MAE explanation impossible. [Investigation summary]({{ '/assets/downloads/lrp-without-labels/mae-findings.md' | relative_url }}).

## A pretty map still owes us an explanation

One useful check is to alter highly ranked regions and measure what happens to the representation. On three images and three unit allocations, replacing the 20 highest-ranked patches with normalized zeros changed MAE's selected output coordinates **less than random patches**: the median top/random effect ratio was **0.915**, with none of nine cases above one. DINOv1 reached **1.314**, with all nine above one.

That is evidence against calling the tested MAE patch ranking successful. It is not a universal victory for DINO. Replacement is a particular intervention, and both models were weak under a different, small-noise patch test. Even MAE's apparent advantage for high-ranked pixels mostly vanished when random controls matched their position within the patch. The control had to account for the very grid we were investigating. [Methods and intervention results]({{ '/assets/downloads/lrp-without-labels/mae-findings.md' | relative_url }}).

I now keep three questions separate when inspecting a map:

1. **Does the implementation do what its rules say?** Check operation coverage, spatial alignment, forward equivalence and accounting.
2. **Is the result numerically dependable?** Check small perturbations, precision changes and large opposing flows.
3. **Does it distinguish and explain the representation we started from?** Compare allocations and test predictions with interventions suited to the claim.

Closing a ledger does not answer all three. In the DINOv2 montage example, only about **0.142** of the initial unit reaches the pixels; prefix tokens receive about **0.008**, and signed local rule gaps account for about **0.850**. Accounting for the difference is useful. Calling the pixel map “one unit conserved” would be wrong.

<details markdown="1">
<summary>How to read the evidence—and reproduce the figures</summary>

All examples here are development data, not a held-out benchmark. The model explorer uses five images and 13 checkpoints. The MAE trace uses three of those images with two raw-mean allocations. It should not be compared numerically with the gallery's uniform-unit allocation. The intermediate-depth and checkerboard galleries reuse all five images and this post's uniform initial relevance, on four and two of the gallery's own checkpoints respectively.

The input is an RGB image resized with Lanczos to a short side of 224 and center-cropped to 224 × 224. Model normalization is checkpoint-specific. Signed pixel views sum RGB relevance; positive-only views average positive RGB channels. Every map uses its full maximum for display scaling, without percentile clipping. The opacity slider only blends the rendering with the photograph.

The downloadable evidence contains measured results, source-report snapshots, raw stage arrays and provenance, so the numerical claims can be checked independently of the pictures. These files are not extra visualization code. Only the compact interactive data is fetched by the page; the detailed reports are optional downloads. See the [file guide]({{ '/assets/downloads/lrp-without-labels/README.md' | relative_url }}).

Download the [interactive data]({{ '/assets/downloads/lrp-without-labels/data.json' | relative_url }}), [input hashes and provenance]({{ '/assets/downloads/lrp-without-labels/provenance.json' | relative_url }}), [recorded model configurations]({{ '/assets/downloads/lrp-without-labels/backbone-heatmaps.json' | relative_url }}) and [export script]({{ '/assets/downloads/lrp-without-labels/export.py' | relative_url }}). The six MAE stage arrays are included as NumPy archives; [the montage/full-mean example is here]({{ '/assets/downloads/lrp-without-labels/mae_misc_all_mean.npz' | relative_url }}). The script extracts recorded arrays and renders them; it does not invent heatmaps or rerun training.

The intermediate-depth and checkerboard figures are the one pair generated directly from this post's own inference, rather than from an archived research snapshot: [this export script]({{ '/assets/downloads/lrp-without-labels/export_intermediate.py' | relative_url }}) runs the same uniform-relevance seeding and composite rules used everywhere else on this page, for four checkpoints spanning both training regimes. See [raw stage arrays]({{ '/assets/downloads/lrp-without-labels/intermediate-stages.npz' | relative_url }}) and the [detailed source report]({{ '/assets/downloads/lrp-without-labels/intermediate-findings.md' | relative_url }}).

The research snapshot is commit `3db2f80`. The archived [experiment guide]({{ '/assets/downloads/lrp-without-labels/experiments.md' | relative_url }}) describes inference reproduction in that repository. Downloaded Markdown reports are source snapshots; their internal relative links refer to the research repository, not this website.

</details>

## What I would use today

I would use these maps to inspect representations, form hypotheses and find suspicious behavior—including background dependence and outliers. I would always keep the starting allocation, signed amounts and propagation rules attached to the picture.

For the tested MAE checkpoint, I would keep the patch-level and pixel-level diagnostics side by side. The former helps locate where a pattern changes; the latter reveals how the projection distributes it. Neither currently earns a claim of validated object-level pixel attribution.

LRP without labels is possible. What disappears is the need for a classifier's answer. What remains is the responsibility to say what amount we started with, how we moved it, and why we trust the resulting trail.

<script src="{{ '/assets/js/posts/lrp-without-labels.js' | relative_url }}" defer></script>

{% comment %}
Parked from the LaFAM post (1,000-image ImageNet box comparison: one val image
per class, 224 centre crop, union of boxes; ResNet-50 supervised / SimCLR /
SwAV; box AUROC = P(inside pixel > outside pixel), ties half, per image).
The LaFAM post keeps only the conclusion that box/mask localization metrics
penalise fine-grained maps. Fold into this post where it fits, or drop.

| Encoder    | LaFAM nearest | LaFAM bilinear | LRP uniform seed | LRP activation seed | LaFAM AnyUp |
| Supervised | 79.25 | 80.66 | 67.27 | 74.79 | 80.23 |
| SimCLR     | 80.83 | 83.20 | 65.67 | 78.37 | 82.41 |
| SwAV       | 79.74 | 82.18 | 63.63 | 73.77 | 80.99 |
Pointing (sup/SimCLR/SwAV): LRP uniform 37.4/49.7/51.8, activation 59.9/80.3/64.8.
Mass inside boxes: activation-seeded LRP has the highest for supervised (68.81).
Median time (RTX 3060 Ti, FP32, single image): LaFAM ~3 ms, AnyUp ~40 ms, LRP ~70 ms.

- The seed matters: activation-proportional instead of uniform seed raises box
  AUROC by ~8-13 points - larger than any gap between the LaFAM variants.
- Uniform seed gives relevance to inactive units: 44-92% of the final feature
  values are zero, depending on the encoder.
- Smoothing inflates the score: on 200 images (every fifth), Gaussian blur
  sigma=16 raises activation-seeded LRP 74.95->79.99 (sup), 78.82->84.07
  (SimCLR), 74.27->79.14 (SwAV); averaging back to 7x7 + bilinear gives about
  the same (79.93 / 83.70 / 78.83).
- Pooled-embedding target (explicit GAP layer inside LRP, then mean): supervised
  uniform-seed 67.37 -> 71.49; no consistent gain for the SSL encoders. Its
  forward scalar equals the spatial mean, but the extra redistribution rule
  changes the map.
- Excluding a 16 px zone around box edges (169 images) does not close the gap:
  activation-seeded LRP stays ~8-14 points below bilinear LaFAM.
- Negative clipping is not the cause: keeping the signed channel mean changes
  uniform-seed LRP by <0.15 points; absolute relevance does not close the gap.
- Toy case (square outline): blurring the perfect map raises box AUROC 71.9 ->
  98.0 while silhouette AUROC falls 100 -> 89.3.
Conclusion then: fine relevance maps and coarse box coverage answer different
questions, and the exact LRP definition (seed, rules) matters.
{% endcomment %}
