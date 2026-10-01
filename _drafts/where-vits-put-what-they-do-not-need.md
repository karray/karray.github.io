---
layout: post
title: Where Does a Vision Transformer Put What It Does Not Need?
tags: XAI, Vision Transformers, Registers, Self-Supervised Learning
description: High-norm tokens, attention sinks and a background that votes for the object, seen through label-free maps and a multi-instance view of classification.
---

<!-- DRAFT. Numbers marked (ours) come from the label-free-maps post and its notebook. Experiments marked TODO are not run yet. -->

In the [previous post]({% post_url 2026-10-01-unsupervised-feature-attribution %}), the label-free maps of three plain vision transformers came out bright almost everywhere. Only a handful of tokens stood out, and they were the darkest spots of all. They sat in the sky above two dogs and on the white background of an image with six objects. Inside the network, however, these same tokens held by far the largest values. Why would a network put its largest activations into a patch of sky?

The effect itself is not new. Darcet et al. found such tokens in several large vision transformers and proposed a remedy, extra tokens called registers [<a href="#darcet2024vision" data-ref="darcet2024vision">Darcet et al.</a>]. What is less settled is why the tokens appear in the first place. This post collects the explanations proposed since then and checks them against what the label-free maps showed. It then adds one more, based on how a classifier treats an image as a bag of patches.

## What the Maps Showed

The dark tokens appeared in a supervised ViT-B/16, a CLIP ViT-B/16 and DINOv2, in every one of our five images. DINOv1, DINOv3 and MAE had none. Neither did the DINOv2 model trained with registers (ours).

Just before the encoder's final layer normalization, each of these tokens has one enormous value. In the supervised ViT, it reaches about a thousand, while the largest value of an ordinary token is a few dozen. The spike always sits in the same channel, channel 187 in the supervised ViT, 474 in CLIP and 415 in DINOv2, whatever the image (ours).

The layer normalization then divides every token by the spread of its values. In a spiked token, that spread comes almost entirely from the one huge channel. All other channels are pushed close to zero, and the token becomes nearly flat. A label-free map averages the positive values of a token, so a flat token comes out dark. A map of token norms taken before the layer normalization would show most of the same tokens as the brightest spots instead.

The dark tokens lay on plain and repetitive background, such as sky and the white background of the image with six objects. None of them fell on the small watermark of the spider web image, which several CNNs responded to strongly (ours).

<!-- TODO: figure. For one image, three panels: label-free map with dark tokens, token norms before the final layer norm, and the value of the spike channel. -->

## Registers and the Memory Explanation

Darcet et al. found high-norm tokens in DINOv2, OpenCLIP and a supervised DeiT-III, but not in the original DINO. The tokens appeared in large models and only after a substantial part of training. They sat in patches that looked much like their neighbours. They kept little information about their own patch and position, but a lot about the image as a whole. Adding a few learnable tokens that are dropped at the output removed them.

Darcet et al. read this as recycling. A patch that repeats its neighbours adds little to the image, so the network can use its token as scratch space for global computation. Registers give the network such scratch space without sacrificing a patch.

Later work has put this reading under pressure. Sun et al. replaced the register states of DINOv2 with their averages over many images, and the classification accuracy barely changed [<a href="#sun2024massive" data-ref="sun2024massive">Sun et al.</a>]. Parodi et al. went further and swapped in noise or registers taken from other images, again with little effect [<a href="#parodi2026zero-ablation" data-ref="parodi2026zero-ablation">Parodi et al.</a>]. Only setting the registers to zero hurt. Jiang et al. traced the spike to a small set of neurons and moved their output into an extra, untrained token at test time [<a href="#jiang2025vision" data-ref="jiang2025vision">Jiang et al.</a>]. This worked about as well as registers learned during training.

The fixed spike channel points the same way. A spike in the same direction in every image says little about any particular image. Whatever image-specific information these tokens carry would have to sit in their other channels. So the high norm itself looks less like memory and more like a constant that the network needs somewhere.

## The Sink Explanation

An attention head mixes the tokens with weights that are positive and sum to one. A head that finds nothing useful to read still has to put its weight somewhere. The network can solve this with a token whose key attracts attention and whose value adds almost nothing. Attention that lands on such a sink is effectively switched off.

Language models build sinks of this kind on the first token and on delimiters such as punctuation [<a href="#xiao2024efficient" data-ref="xiao2024efficient">Xiao et al.</a>, <a href="#bondarenko2023quantizable" data-ref="bondarenko2023quantizable">Bondarenko et al.</a>]. Bondarenko et al. traced large activation outliers in BERT and in vision transformers to heads that try to do nothing, and removed them by changing the softmax or gating the attention output [<a href="#bondarenko2023quantizable" data-ref="bondarenko2023quantizable">Bondarenko et al.</a>]. Gu et al. found that sinks behave like a bias on the keys, and that language models with unnormalized attention do not form sinks, at least up to a billion parameters [<a href="#gu2025when" data-ref="gu2025when">Gu et al.</a>]. Sun et al. showed that the massive activations act as fixed biases in both language models and vision transformers.

This explains the fixed channel well, since a sink should look the same in every image. It also suggests why sinks land on plain background. Giving up a patch of sky costs the network little. It explains less well why DINOv1 and MAE have no such tokens, although they use the same softmax attention.

## A Classifier Sees a Bag of Patches

There is a second question hidden in the dark tokens, and it concerns the rest of the background. In my recent work, I looked at classifiers that average their spatial features before a linear head [<a href="#karjauv2026rethinking" data-ref="karjauv2026rethinking">Karjauv</a>]. For such a classifier, the score of the image is exactly the average of the scores of its patches.

$$
z = \frac{1}{N} \sum_{i=1}^{N} \left( W h_i + b \right)
$$

Each patch is an instance, and the image is a bag of instances with one label. A patch has no way to abstain, because the classifier has no background class. Every patch of sky or grass must vote for one of the classes the model knows. The background usually covers most of the image, so it could easily outvote a small object.

There are two ways to keep that from happening. The network can keep the background quiet, so that its votes barely move the average. Or it can give the background the object's identity, so that the background votes for the object too. A CNN cell sees only its own neighbourhood and cannot know what the object is, so only the first way is open to it. Attention lets every token read from the object, so the second way becomes available.

The data fit this picture. In about nine out of ten images, Swin and MaxViT predict the target class in at least one background cell. Once the object is hidden, this falls below four in ten. For ResNet-50, the rate starts at about one in three and drops only slightly. A plain ViT that classifies from its class token shows the same pattern, from 86 to 24 percent, even though its patch tokens never reach the classifier directly. The label-free maps of the previous post told a similar story. The background of a ResNet stayed quiet, while in the supervised ViT and CLIP, the channels that never switch off were strongest on the background.

Shi et al. describe the same diffusion for models that classify from a class token, and call it lazy aggregation [<a href="#shi2026vision" data-ref="shi2026vision">Shi et al.</a>]. The class token gathers the image's meaning from background patches that already carry it, instead of from the object. Registers do not prevent this, and the dense features of DINOv3 still degraded during long training until the authors added an extra loss to anchor them [<a href="#simeoni2025dinov3" data-ref="simeoni2025dinov3">Siméoni et al.</a>].

A bag can also hold several valid objects. In a synthetic experiment, each image contained three to five objects but was labeled with only one of them. The models still learned to assign the correct class to each object in their patch scores. Their image-level accuracy stayed where it would be if they picked one of the visible objects at random, which is the best any model can do with such labels.

## Two Questions, Two Answers

Three things happen to a token in a vision transformer, and they are easy to mix up.

| What happens | Likely cause | Removed by registers? |
|---|---|---|
| A token takes in information from the rest of the image and still describes its own patch | ordinary attention | not a problem |
| Background tokens take on the meaning of the object | global attention and an image-level objective | no |
| A few tokens become sinks with one enormous value | softmax attention, which cannot attend to nothing | yes |

Seen side by side, the last two look like answers to the same question. Where does a token put what it does not need? Inside attention, the softmax forces every head to put its weight somewhere, and the network builds sinks. Inside the label space, the classifier forces every patch to vote for some class, and the network recruits the background. Registers give the first answer a place of its own. Nothing plays that role for the second, since a standard classifier has no background class. This is a hypothesis, not a finding.

Could the objective also shape the sinks? MAE reconstructs pixels and has no image-level target, and it has no outlier tokens in our images. DINOv1 does have an image-level target and has none either. The objective alone therefore does not decide whether sinks appear.

## What Remains Open

Do the dark tokens receive most of the attention in the late layers, while adding little through their values? That is what a sink should do, and it can be measured directly on the same five images.

<!-- TODO: attention mass on outlier tokens per layer and head, and the norms of their value vectors. -->

Would a vision transformer trained on bags with several labeled objects recruit less background than one trained with a single label? The synthetic dataset allows exactly this comparison, since every object in it is annotated.

<!-- TODO: ViT on the synthetic data, single-label vs multi-label loss. Compare background activation after occlusion, and whether high-norm tokens appear at all. Caveat: Darcet et al. saw them only in larger, longer-trained models. -->

Why do DINOv1 and MAE avoid outlier tokens, while DINOv2, CLIP and the supervised ViT do not? All of them use the same attention.

Do sinks prefer plain patches or repetitive ones? In natural images the two usually coincide, since a sky is both.

<!-- TODO: synthetic controls. Repeated textured tiles (repetitive, not plain), a small plain patch in a busy scene (plain, not repetitive), a logo-like watermark. -->

## References

{%- capture references -%}
Darcet, T., Oquab, M., Mairal, J., & Bojanowski, P. | 2024 | Vision Transformers Need Registers;
Sun, M., Chen, X., Kolter, J. Z., & Liu, Z. | 2024 | Massive Activations in Large Language Models;
Parodi, F., Matelsky, J., & Segado, M. | 2026 | Zero-Ablation Overstates Register Content Dependence in DINO Vision Transformers;
Jiang, N., Dravid, A., et al. | 2025 | Vision Transformers Don't Need Trained Registers;
Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. | 2024 | Efficient Streaming Language Models with Attention Sinks;
Bondarenko, Y., Nagel, M., & Blankevoort, T. | 2023 | Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing;
Gu, X., et al. | 2025 | When Attention Sink Emerges in Language Models: An Empirical View;
Karjauv, A. | 2026 | Rethinking Global Average Pooling: Your Classifier Is Secretly a Multi-Instance Learner;
Shi, C., et al. | 2026 | Vision Transformers Need More Than Registers;
Siméoni, O., et al. | 2025 | DINOv3;
{%- endcapture -%}

{% include reference.html ref=references %}
