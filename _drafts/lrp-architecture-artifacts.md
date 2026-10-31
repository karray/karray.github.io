---
layout: post
title: Where Do the Grids in LRP Maps Come From?
tags: XAI, LRP, CNN, ViT, Self-Supervised Learning
description: Some patterns in relevance maps come from the architecture rather than the image, such as a checkerboard in ResNet and a patch grid in MAE.
---

Relevance maps of some vision models carry patterns that have nothing to do with the image. ResNet's intermediate maps show a checkerboard, and the pixel maps of MAE show a grid of patch edges. This post traces both back to the architecture.

<!--more-->

## What happens in between is not just a blurrier version of the ends

Every map so far has been an input-level pixel map or a starting embedding. None of them show what relevance looks like partway through the network. For a stack of convolutions, that turns out to matter: one architecture carries a real, structural checkerboard through its intermediate stages, and another does not. To check that this is about the architecture and not about how the relevance got started, everything below uses the same uniform, label-free allocation as the rest of this post—on a supervised **and** a self-supervised checkpoint of each architecture.

Choose any of the five images, then switch between ResNet50 and ConvNeXt-B, each supervised and each trained with DINO, at each stage output from right after the stem to the last stage before pooling.

{% include figure-gallery.html id="lrp-intermediate-maps" data=site.data.galleries.lrp_intermediate %}

Positive channel mean, clipped and scaled independently at each depth: black means small or nonpositive relevance, not necessarily zero. Depth 4 is where the uniform initial relevance itself lives—every map there is flat by construction; the interesting question is what a network does to it on the way back to depth 0.

A **phase ratio** compares the strongest and weakest of the four spatial phases a stride-2 layer produces (even/even, even/odd, odd/even, odd/odd); 1.0 means no imbalance. Both ResNet checkpoints start close to **9** right after the stem and stay above **4.7** through the middle stages—supervised and self-supervised alike. Both ConvNeXt checkpoints never leave **1.0–1.2**. Training objective does not decide which architecture shows this; the architecture does.

### Two branches, opposite phase

ResNet's imbalance has a mechanism, not just a measurement. Every stage transition (for example `layer4.0`) adds a **main branch**, a 3 × 3 stride-2 convolution, to a **shortcut**, a 1 × 1 stride-2 convolution. Isolate each branch's relevance at the transition it feeds, again on any image and for both the supervised and the DINO checkpoint, and compare an unsplit residual against one split by activation magnitude:

{% include figure-gallery.html id="lrp-checkerboard-maps" data=site.data.galleries.lrp_checkerboard class="wide" %}

Absolute channel-summed relevance; independent full-range scales.

The main branch's 3 × 3 windows are centered on even positions, so an odd/odd position sits inside four overlapping windows against one for even/even. The shortcut's 1 × 1 window cannot overlap at all—it returns relevance **only** at even/even positions; the other three quarters of positions get exactly zero from it, a genuine disconnection rather than noise, not something a different rule could route around. Both patterns are structural, not content-dependent: they show up identically under unit activations and uniform incoming relevance, with no photograph involved.

The visible checkerboard is their weighted sum, and here is a wrinkle worth being honest about: this post's own production rule for splitting a ResNet residual—**equal**, a fixed 50/50—turns out not to change that sum at all. Dividing both branches by the same constant leaves their *relative* weighting exactly where it was; it is mathematically indistinguishable from not splitting. Only a rule that treats the branches asymmetrically moves anything. Switching to a **magnitude**-weighted split—each branch's share proportional to its own activation size at that position—lowers the even/even share from **0.571** to **0.429** on the supervised checkpoint and from **0.560** to **0.374** on the DINO checkpoint. Real, and the same direction for both training regimes, but a rule this post does not actually use anywhere else.

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

{% include figure-gallery.html id="lrp-projection-maps" data=site.data.galleries.lrp_stems %}

On this montage, the fraction of within-patch map energy captured by one repeated template drops from **99.6% to 82.8%** for DINOv2 and **99.3% to 84.0%** for DINOv3. That explains a visible change in texture. It does not prove that ZPlus identifies the right pixels, and it does not eliminate all patch structure.

There is also unfinished business upstream: different DINOv2 output-channel allocations produced strongly correlated pixel maps. A detailed-looking map can lose distinctions that were present in the representation. Better texture and better specificity are separate achievements.

## MAE: the grid has an accomplice

[Masked autoencoders](https://arxiv.org/abs/2111.06377) learn by reconstructing missing image patches. Here we use the trained encoder on the complete image, with no random masking and no decoder. [DINO](https://arxiv.org/abs/2104.14294), by contrast, uses self-distillation; its authors reported emergent spatial structure in self-supervised ViTs. Different learning objectives make different representations plausible. They do not, by themselves, explain our attribution results.

MAE's projection weights provide a more concrete clue. A patch's outermost rows and columns occupy **23.44%** of its area, but contain **54.30%** of MAE's squared projection weights. The tested DINOv1 and supervised ViT checkpoints are near the area fraction: **22.55%** and **23.15%**.

This boundary preference also appears outside LRP. Native gradient energy concentrates at patch edges; matched perturbations to boundary pixels change MAE's embeddings more than equally sized, equal-energy interior perturbations in the tested pairs. The grid has a model-dependent component. We have not established which part of training caused it.

But where does the grid enter the explanation? Follow an actual backward trace. Start at the final embedding, then move toward the pixels. Compare the last two stages: the pixel map and those same pixel amounts added back together within each original patch.

{% include figure-gallery.html id="lrp-backward-maps" data=site.data.galleries.lrp_trace %}

Blue is negative, red positive. Each stage has its own symmetric full-range color scale. Coarse panels are actual 14 × 14 channel sums enlarged without smoothing; they are not pixel explanations or selected-patch masks.

Across these six cases, the incoming patch amounts and the pixel amounts summed back within their patches correlate **0.994–0.9999**. Much of the conspicuous fine grid arises *inside* the patches. Coarse structure can survive while the pixel redistribution looks unhelpful. That still leaves the coarse ranking itself to validate.

### A negative mean is not automatically “evidence against the object”

MAE's final normalization includes a learned bias. Averaged over coordinates, that bias is **−0.03298339**, the same for every image. On the dog/cat input, the image-dependent term is about **+0.02331165**. Add the bias and the full embedding mean is **−0.00967174**.

The negative total does not mean the dog and cat are negative concepts. It partly reflects where the model places the origin of its feature coordinates. The exact output decomposition above exposes that offset. We keep the initial relevance unchanged and account for parameter contributions separately; we do not flip the finished heatmap to make it look nicer.

For positive-only means, the set of participating coordinates changes with the image. Their bias contribution need not be a constant spatial offset. Switch the allocation in the trace to see the measured decomposition change.

<details markdown="1">
<summary>Why not rescale embeddings to −1…+1 first?</summary>

That defines another allocation. It may be useful if its meaning fits the question, but it is not neutral housekeeping. In our three MAE examples, a global min–max transform changed the fraction of positive coordinates from roughly 46–47% to **0.362%, 99.351% and 99.759%**. One extreme coordinate can move the zero point for almost everything else.

Similarly, stabilizing a denominator can change the attribution rather than merely compute it more accurately. Increasing the final normalization stabilizer improved MAE's numerical robustness, yet made different channel explanations more alike and weakened one patch-replacement test. Stability is necessary; it is not sufficient.

</details>

## Could we just take the shortcut?

A transformer has a residual stream carrying information around its attention and feature-mixing blocks. It was tempting to route relevance through that stream and bypass the difficult operations.

We tried seven backward configurations on three images and both embedding means: 42 cases. Identity-only routing, skipping attention, skipping feature mixing, equal splits and other alternatives all retained the pixel grid. The forward network was identical; these were changes to the attribution assumptions, not models with layers removed.

There is a reason the shortcut cannot settle the question. Preserving an amount on one branch tells us nothing about whether assigning zero to the other branch is justified. And none of those routes avoids the final step from patch features into pixels.

Other tempting improvements failed for different reasons. Selecting object-like patch groups largely reproduced the selection itself. Averaging explanations from shifted images reduced grid visibility, but explained multiple altered inputs and did not establish better faithfulness. Changing the pixel rule to respect physical RGB bounds improved invariance to coordinate conventions, yet kept the grid. These are informative failures, not grounds for declaring every future MAE explanation impossible.
