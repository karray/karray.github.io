---
layout: post
title: "Figure lab: dense readouts"
description: "Working page for the figures of the MIL post."
tags: Draft
---

Working page, not for publication. Every figure uses data from `gap_mil_revision/scripts/blog_readouts.py` (four of the paper's ImageNet classifiers with their exact readouts, centre crop, the sample images); each has a model menu. All readouts share one layout: the image-level answer on the left, the image with each cell's top class in the middle, the cells on the right.

<!--more-->



## Recap figure

The relevance-flow figure of the LRP post, with ResNet-50 selected first. The LRP post keeps its own default.

{% include posts/label-free-lrp-stack.html id="lab-flow" model="resnet50" %}

## 1. Dense readout

Each cell shows its top class. Hover or tap cells for their top 5, pick a class to see its score in every cell; the card under the image list then states their average, which is the image score.

{% include posts/gap-mil-readout.html id="lab-readout" variant="readout" items="pets_indoors:Pets indoors;six_objects:Six objects;two_dogs:Two dogs;dog_and_cat:Dog and cat;bus_and_taxi:Bus and taxi;doughnuts_newspaper:Doughnuts;spider_web:Spider web" %}

## 2. Shortcuts: the object removed

Three pairs: the tench with the fish inpainted away, the guardsman without the bearskin hat (synthetic), and the arms without the dumbbells (synthetic, a control). Hover or tap a cell to see its top 5; cells cannot be picked here.

{% include posts/gap-mil-readout.html id="lab-pair" variant="pair" labels="Object present|Object removed" items="tench|tench_removed:Tench;bearskin|bearskin_removed:Bearskin;dumbbell|dumbbell_removed:Dumbbell" %}

## 3. Spillover across layers (CNNs and ViTs section)

The warthog and the same picture with the cells under the animal greyed out (cell-aligned occlusion as in the paper: every cell touching the animal filled with the ImageNet mean colour). Read out after intermediate blocks: the final norm and the classifier applied wherever the path from a block to the head acts on each cell alone. ViT-M/16 GAP: every one of its 12 blocks, on its 16 × 16 token grid (the same view at the same scale, so each 7 × 7 cell is 2 × 2 tokens and the occlusion is token-aligned). Swin-B: every second block of stage 3, through its patch merging, and stage 4. ConvNeXt-B: every third block of stage 3, through its 2 × 2 downsampling, and stage 4. Top row of the strip: the original, bottom row: the occluded picture. Red cells vote *warthog*.

{% include posts/gap-mil-readout.html id="lab-layers" variant="pair" labels="Original|Occluded" models="vitm:ViT-M/16 GAP;swin:Swin-B;convnext:ConvNeXt-B" items="warthog_layers|warthog_occluded_layers:Warthog" %}

### 3B. Occlusion, final layer

The same pair read out at the final layer only, for the four models of the post.

{% include posts/gap-mil-readout.html id="lab-cover" variant="pair" labels="Original|Occluded" image="off" items="warthog|warthog_occluded:Warthog" %}

## 4. Adversarial attack

Targeted PGD towards *sombrero*, 20 steps, L-infinity budget 8/255. The strip of small grids is the slider.

{% include posts/gap-mil-readout.html id="lab-attack" variant="attack" items="dog_and_cat_pgd:Dog and cat" %}
