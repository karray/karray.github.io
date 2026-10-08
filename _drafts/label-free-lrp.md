---
layout: post
title: Following Relevance Through Modern Vision Models
styles:
  - /assets/css/posts/label-free-lrp.css
tags: XAI, LRP, CNN, ViT, Self-Supervised Learning
description: How we wrote LRP rules for modern vision encoders without labels, what broke along the way, and what it tells us about how these models work inside.
---

Layer-wise relevance propagation (LRP) is a powerful tool to understand what a neural network looks at when it makes a prediction. It sends the model's output back through the network, layer by layer, until it arrives at the pixels. The trick is to choose the right rules for each layer, so that the relevance is divided among the inputs in proportion to their contribution. Without the right rules, the maps dissolve into noise. This post investigates missing rules for modern vision encoders, and what we ca learn about them along the way.

<!--more-->

Think of the output of a network as an amount of water that we pour back through it. At every layer, the water is divided among the units that fed into the current one, and a unit that contributed more receives more. When the water reaches the image, the amount on each pixel tells us how much that pixel contributed. This amount is called relevance, and the procedure is layer-wise relevance propagation (LRP) [<a href="#bach2015on" data-ref="bach2015on">Bach et al.</a>].

{%- comment -%}TODO: visualization for water analogy based on the original LRP illustration{%- endcomment -%}.

In the previous post, we used LRP to explain self-supervised encoders without relying on labels. However, [zenit](https://github.com/zenit/zenit), a library for implementing LRP in PyTorch, only covers convolutional networks such as ResNet and VGG. We therefore had to write the missing rules for the blocks in ConvNeXt and transformer-based vision models.

This post follows the relevance through these models one building block at a time. For each block, a small interactive figure shows what goes wrong without a rule and what the rule does instead. Along the way, we point to published work that arrived at the same or better solutions. The rules and all examples are available in the [companion notebook](https://github.com/karray/deep-unboxing) {%- comment -%}TODO: link to 02_... once it exists{%- endcomment -%}.


The figure below shows this inside two real encoders. On the left is where the relevance starts, the feature grid at the end of the network. To its right is the same relevance after it has been passed back to ever earlier layers, until it arrives at the pixels.

{% include posts/label-free-lrp-stack.html %}


Note that that the cat gets less relevance than the dog. The reason that the final relevance is relative and since the dog has a larger activation it gets more relevance, while cat's relevance is downscaled.

How the relevance is divided at each layer is decided by a rule. Each kind of layer needs a rule that fits what it computes, and choosing them well is most of the work [<a href="#montavon2019layer-wise" data-ref="montavon2019layer-wise">Montavon et al.</a>]. This is where our trouble began.

One more choice concerns the start. LRP was designed to explain a prediction, so it usually starts from a class score. A self-supervised encoder has no class score. We start from its features instead and use only their positive part. In the [previous post]({% post_url 2026-10-01-unsupervised-feature-attribution %}), we averaged the same positive features into a coarse map, and LRP traces that map back to the pixels.

<details markdown="1">
<summary>Why only the positive part of the features?</summary>

The features of most modern encoders are signed. If we started from the raw values, positive and negative values would largely cancel. In ConvNeXt, Swin and the plain ViTs, the sum of all feature values keeps only 1 to 5% of their total magnitude, and for several of these models it is negative for almost every image. ResNets do not have this problem, because their features pass through a ReLU and are never negative. Using the positive part avoids the cancellation and gives inactive features no relevance at all.

</details>

## Adding the Rules One at a Time

Two libraries provide ready-made LRP rules for PyTorch, and both come from the group at Fraunhofer HHI and TU Berlin that introduced LRP. [zennit](https://github.com/chr5tphr/zennit) covers convolutional networks such as VGG and ResNet [<a href="#anders2021software" data-ref="anders2021software">Anders et al.</a>]. [LXT](https://github.com/rachtibat/LRP-eXplains-Transformers), the implementation of AttnLRP, covers language models and the original vision transformer [<a href="#achtibat2024attnlrp" data-ref="achtibat2024attnlrp">Achtibat et al.</a>]. Their rules for the layers with weights, the convolutions and linear layers, carry over to other models. Our encoders combine these layers with LayerNorm, GELU, attention and residual connections in their own ways, and none of them was covered.

So we started where the libraries stop. We gave the weighted layers their usual rules and left everything else alone. An operation without a rule passes relevance back the same way it passes a gradient. The first step in the figure below shows the result. For all three models, the map is noise.

{% include figure-gallery.html id="lrp-rules" data=site.data.galleries.lrp_rules %}

Over the next sections, we add the missing rules one at a time, in the same order as the slider.

### Residual Connections

Most modern blocks do not replace their input. They compute an update and add it to the input, which is called a residual connection. Going backward, the relevance of the sum has to be shared between the input, which arrives through the skip path, and the update.

The gradient of an addition passes the full amount to both sides. An addition without a rule therefore copies the relevance instead of dividing it. The figure below shows a single addition. In the first row, both sides receive the full amount, and the total doubles.

<figure class="lrp-fig">
  <p class="lrp-kicker">One residual connection</p>
  <div data-lrp-toy="residual" data-rules="copy,equal,proportional" data-presets="false"><p class="lrp-small">This figure needs JavaScript.</p></div>
  <figcaption>Move the sliders to change the two values that are added. The rows show how much relevance each side receives when the output holds a relevance of 1. Purple bars are negative.</figcaption>
</figure>

With every block copying, the amounts grow quickly. In the DINOv1 ViT, the pixels receive more than a thousand times the starting amount. We therefore rewrite every addition so that it can receive a rule. In the second step of the gallery above, every addition splits the relevance between its two sides. The amounts become plausible, but the maps are still noise. In Swin, they even grow to millions at this step, until LayerNorm receives its rule as well. Which split to choose turned out to be a question of its own, and we return to it later.

### LayerNorm

LayerNorm takes the values of a token, subtracts their mean and divides them by their spread. Adding the same number to every input therefore changes nothing at the output. Because of this, the gradient of LayerNorm always sums to zero over its inputs. Relevance that passes through LayerNorm without a rule is split into positive and negative parts that cancel exactly. Try it below. However you set the inputs, the total in the last column stays at zero.

<figure class="lrp-fig">
  <p class="lrp-kicker">One LayerNorm</p>
  <div data-lrp-toy="layernorm"><p class="lrp-small">This figure needs JavaScript.</p></div>
  <figcaption>Six input values are normalized. The positive outputs receive a relevance of 1 in total, in proportion to their size, just like the start of our explanations. The two right columns show what each input receives with our rule and without any rule. The buttons shift or stretch all inputs at once, which leaves the output unchanged.</figcaption>
</figure>

Every block of our encoders contains at least one LayerNorm. Each of them turns the relevance into a balanced mix of positive and negative amounts, which is why the maps looked like noise. Our rule holds the spread fixed during the backward pass, while the forward pass stays exactly the same. LayerNorm then behaves like a linear layer, and relevance passes through it with the usual rule for linear layers. In the third step of the gallery, the objects appear in all three models. Conservative propagation proposed this treatment for transformers [<a href="#ali2022xai" data-ref="ali2022xai">Ali et al.</a>], and AttnLRP uses an equivalent rule [<a href="#achtibat2024attnlrp" data-ref="achtibat2024attnlrp">Achtibat et al.</a>].

### Activation Functions

Activation functions such as GELU change each value on its own. The usual LRP treatment passes relevance through them unchanged, which AttnLRP calls the identity rule [<a href="#achtibat2024attnlrp" data-ref="achtibat2024attnlrp">Achtibat et al.</a>]. Without a rule, the relevance is multiplied by the slope of the function, as a gradient would be. For most inputs, the slope of GELU is below one, so some relevance is lost at every activation.

<figure class="lrp-fig">
  <p class="lrp-kicker">One activation function</p>
  <div data-lrp-toy="activation"><p class="lrp-small">This figure needs JavaScript.</p></div>
  <figcaption>The GELU curve and its slope at the chosen input (dashed). The bars show the relevance that one value passes back when it holds a relevance of 1.</figcaption>
</figure>

zennit applies the identity rule to the activation functions it knows. The ConvNeXt we used implements GELU with its own class, which zennit does not recognize. Nothing fails, and in the fourth step of the gallery the ConvNeXt map barely changes. Yet the amount reaching the pixels rises from 0.48 to 0.66 for the dog and the cat. Without the rule, a quarter of the relevance disappeared without a trace. This was the first lesson of the project. A plausible map is no proof that every layer was handled.

### Attention

Attention lets every token collect information from the other tokens. Each token takes a weighted average of what the other tokens send, and the weights are computed from the tokens themselves. Without a rule, relevance follows the gradient. Each source token then receives relevance in proportion to its weight, no matter how much it actually sent. The rest flows into the computation of the weights, in positive and negative amounts that cancel. We hold the weights fixed during the backward pass instead, so that relevance follows what each token actually sent.

<figure class="lrp-fig">
  <p class="lrp-kicker">One attention step</p>
  <div data-lrp-toy="attention"><p class="lrp-small">This figure needs JavaScript.</p></div>
  <figcaption>An output token averages the values of four source tokens. Each source has a value (what it sends) and a score, and the scores are turned into weights that sum to 1. The output holds a relevance of 1. The two right columns show what each source receives with the weights held fixed and without any rule.</figcaption>
</figure>

This is the approach of conservative propagation [<a href="#ali2022xai" data-ref="ali2022xai">Ali et al.</a>]. AttnLRP goes further and also passes relevance through the computation of the weights, with rules designed for it, which made its explanations of classifiers more faithful [<a href="#achtibat2024attnlrp" data-ref="achtibat2024attnlrp">Achtibat et al.</a>]. In our encoders, holding the weights fixed changed the maps only slightly, as the last step of the gallery shows.

Attention also explains a difference you can see in the first figure. A convolution only combines neighbouring cells, so relevance in ConvNeXt stays close to where it started. Attention connects every token with every other one, so in the ViT, relevance can jump across the image within a single block.

### Smaller Pieces

A few smaller parts needed attention as well. Some blocks multiply their update by a learned constant (layer scale), and MaxViT gates its channels with squeeze-and-excitation. Both simply pass relevance on. Batch normalization can be folded into the convolution before it, which turns two layers into one and leaves the forward pass unchanged. zennit calls such rewrites canonizers. Folding is only exact when the convolution feeds nothing else, so we check that before merging.

{%- comment -%}TODO: cite canonization / batch-norm folding work (e.g. Motzkus et al. 2022; Hui and Binder 2019) after verifying.{%- endcomment -%}

<details markdown="1">
<summary>All rules at a glance</summary>

<div class="lrp-scroll" markdown="1">

| | ResNets | ConvNeXt | Swin | MaxViT | Plain ViTs |
|---|---|---|---|---|---|
| first layer | z+ | WSquare | WSquare | WSquare | z+ |
| layers inside blocks | z+ | Gamma | Gamma | Gamma | Gamma |
| activations, layer scale, gates | pass | pass | pass | pass | pass |
| LayerNorm | — | spread held fixed | spread held fixed | spread held fixed | spread held fixed |
| attention | — | — | weights held fixed | weights held fixed | weights held fixed |
| residual split | equal | proportional | proportional | magnitude | magnitude |

</div>

The Gamma rule favors positive contributions and makes maps less noisy. AttnLRP recommends it for the linear layers of vision transformers [<a href="#achtibat2024attnlrp" data-ref="achtibat2024attnlrp">Achtibat et al.</a>]. The first-layer rules and the residual splits are discussed below.

</details>

## How We Checked Our Rules

Since a broken rule can still produce a plausible map, we relied on a few simple checks rather than on the pictures. They are worth running for any model one adds.

- The model's output must not change when the rules are installed. Every rewrite only changes the backward pass.
- Every layer must receive a rule. Counting the layers that have no rule catches cases like the GELU above.
- The amount reaching the pixels should be plausible. It should not exceed the starting amount by orders of magnitude.
- After the explanation, the model must be restored exactly, so that later predictions are not affected.

We run LRP on the CPU. It takes a few seconds per image and gives identical numbers on every machine.

{%- comment -%}TODO: these checks come from the research code; add them to notebook 02 (and xai_utils) so the claim holds for the published package.{%- endcomment -%}

## Who Gets the Credit at a Residual Connection

At a residual connection, the output is the sum of two parts. One part is the input passed along unchanged, the skip. The other is what the block adds, the update. Going backward, LRP must decide how much of the relevance each part receives. There are several reasonable answers, and they behave very differently.

- An **equal** split gives each part half, regardless of the numbers.
- A **proportional** split gives each part what it contributed. This is the standard LRP treatment of an addition, and it is what LXT and AttnLRP do implicitly. A part that worked against the output receives negative relevance.
- A **magnitude** split divides by size and ignores the sign. We introduced it for MaxViT and the ViTs, after the proportional split made some of their maps unstable in our early experiments.
- A **γ split** works like the proportional one, but gives extra weight to the part whose sign agrees with the output. It was proposed by Berend et al. while we were writing this post [<a href="#berend2026reslrp" data-ref="berend2026reslrp">Berend et al.</a>].

<figure class="lrp-fig">
  <p class="lrp-kicker">Four ways to split</p>
  <div data-lrp-toy="residual" data-rules="equal,proportional,magnitude,gamma"><p class="lrp-small">This figure needs JavaScript.</p></div>
  <figcaption>The same residual connection as before, now with the four splits. The γ split uses γ = 1. The buttons set typical cases.</figcaption>
</figure>

When both parts push in the same direction, the rules agree, except for the equal split. When the update works against the skip, they disagree. The proportional split keeps the information that the update worked against the output, but its numbers grow without bound as the two parts cancel. The magnitude split never grows, but it credits the opposing update as if it had helped. The γ split keeps the sign and is guaranteed to stay bounded. Berend et al. identify such cancellation inside transformer blocks as a main reason why LRP maps of ViTs become noisy.

{% include figure-gallery.html id="lrp-residual" data=site.data.galleries.lrp_residual %}

In the maps, the equal split stands out first. In a deep network, it sends half of the relevance into every block's update, where part of it is absorbed. Very little reaches the pixels, as little as 0.04 for the DINOv1 ViT. The proportional split delivers the most relevance, but in DINOv1 it also spreads it over the grass and the trees. The magnitude and γ splits keep more of it on the animals. The γ split passes on more relevance than the magnitude split and is a little more detailed.

We chose our rules by looking at maps like these and by checking that small changes to the input did not change the maps much. We did not test which rule is more faithful to the model, and on a handful of images we would not want to claim one. For anyone who wants to choose between them, Berend et al. compare the γ split with the standard rules on several faithfulness measures.

<details markdown="1">
<summary>A few numbers behind the maps</summary>

On eight ImageNet images per model, we measured three things for each rule. The first is how much relevance reaches the pixels. The second is how much the map changes when the input moves by a tiny amount. The third is whether the map follows its start, which we tested by starting only from the feature cells inside the object box and then only from those outside.

The equal split delivered the least relevance (0.00 to 0.12 for most models) and its maps barely depended on where they started. The proportional split delivered the most (up to 0.97) and followed its start most closely, and it stayed stable on these images. The magnitude and γ splits were the most stable. The γ split followed its start almost as closely as the proportional one. The maps of the supervised ViT barely depended on their start under any rule.

</details>

## Where the Relevance Goes

In the first figure, the bands narrow on the way to the pixels. How much arrives differs a lot between models.

| Model | Relevance at the pixels |
|---|---|
| ResNet-50 | 0.02 |
| ConvNeXt | 0.53 |
| Swin | 0.54 |
| MaxViT | 0.13 |
| ViT-B/16 | 0.85 |
| DINOv1 ViT | 0.15 |
| DINOv3 ViT | 0.30 |
| MAE | 0.30 |

{%- comment -%}Median over 8 ImageNet images, tutorial rules, start = 1. Source: lrp/temp/lrp_post/NOTES.md.{%- endcomment -%}

The rest stays inside the network. Most layers add a constant, the bias, to their output. The bias is not an input, yet part of the output is due to it, so the rules leave part of the relevance with it. Small stabilizing terms in the rules keep a little more. In transformers, the class token and the register tokens take their share as well, because they are learned parameters rather than parts of the image.

Is this a flaw? AttnLRP argues that it is not. Following the original LRP paper, it treats a bias as an extra input whose value is always one, so that its relevance is "preserved, rather than lost" [<a href="#achtibat2024attnlrp" data-ref="achtibat2024attnlrp">Achtibat et al.</a>]. One could force all relevance onto the pixels, by spreading the bias's share over the inputs or by leaving the bias out of the rules. AttnLRP shows that both options can make the results numerically unstable or flip their signs. In our own tests, leaving out the bias kept all of the relevance in a ResNet but produced invalid numbers in ConvNeXt. We accept that part of the relevance stays inside the network and report how much reaches the pixels.

Position embeddings are the one place where our rules currently err in the other direction. A ViT adds them to the patch tokens with a plain addition, so the patches receive the share that belongs to the positions. Bakish et al. show that position embeddings carry relevance of their own and that ignoring them violates conservation [<a href="#bakish2025revisiting" data-ref="bakish2025revisiting">Bakish et al.</a>].

## The Last Step to the Pixels

The first layer of many modern encoders cuts the image into small patches and turns each patch into a feature vector, four by four pixels in ConvNeXt and Swin and sixteen by sixteen in a ViT. Its rule decides how a patch's relevance is spread over its pixels, and with that, most of the fine detail in the map.

- **z+** spreads relevance according to the positive contributions of the actual pixel values.
- **WSquare** spreads it according to the squared weights alone and ignores the pixel values. It is recommended for inputs without natural bounds [<a href="#montavon2019layer-wise" data-ref="montavon2019layer-wise">Montavon et al.</a>].
- **Gamma** is the rule we use inside the blocks.

{% include figure-gallery.html id="lrp-stems" data=site.data.galleries.lrp_stems_rules %}

Look closely at the maps, and the borders between the patches become visible. WSquare paints visible edges along them, because it gives every patch the same pattern regardless of its content. z+ shows fewer edges, and Gamma almost none. The number under each map measures this as the contrast between neighbouring pixels across a patch border and within a patch, where 1 means no edges. Over the five images, the median for ConvNeXt is 2.11 with WSquare, 1.66 with z+ and 0.97 with Gamma. Swin and the DINOv1 ViT behave the same way. The rules in our table are the ones we settled on early, WSquare for ConvNeXt, Swin and MaxViT and z+ for the ViTs. These measurements suggest that Gamma deserves a closer look.

MAE is the exception. Its maps keep a strong grid under every first-layer rule, and Gamma does not remove it. The grid was also visible in the previous post, where it dominated MAE's maps. Where it comes from is a question of its own, and we look at it in a separate post.

## What We Did Not Solve

Several questions remain open.

- Only part of the relevance reaches the pixels, and how much differs a lot between models and rules.
- The maps depend on the rules. The residual split alone changes how much relevance arrives and where it goes.
- In plain ViTs, the maps barely depend on where the relevance starts. For the supervised ViT, this held under every residual rule we tried. If the start hardly matters, it is not obvious what exactly such a map explains.
- We did not measure faithfulness. Readers who need it will find established measures in the work of Achtibat et al. and Berend et al.

## Use It Yourself

All rules live in a small package that comes with the notebook. Explaining an encoder takes a few lines.

```python
from xai_utils.models import load_model
from xai_utils.lrp import explain

model = load_model("dino_vit")              # one of the thirteen encoders
x = model.transform(image)[None]            # a 224x224 PIL image -> (1, 3, 224, 224)
relevance = explain(model, x)               # (3, 224, 224), one map per colour channel
heatmap = relevance.clamp(min=0).mean(0)    # positive part, averaged over the colours
```

To add a new model, one writes a canonizer for each of its blocks that makes the residual additions explicit and holds the attention weights and the LayerNorm spread fixed. The checks above then show whether anything was missed.

## Conclusion
{%- comment -%}TODO: write once the sections are agreed. Open questions grounded in the figures: why ViT maps
barely depend on their start; which residual split to trust; where the grids come from (next post).{%- endcomment -%}
todo


## References

{%- capture references -%}
Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K.-R., & Samek, W. | 2015 | On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation;
Montavon, G., Binder, A., Lapuschkin, S., Samek, W., & Müller, K.-R. | 2019 | Layer-Wise Relevance Propagation: An Overview;
Anders, C. J., Neumann, D., Samek, W., Müller, K.-R., & Lapuschkin, S. | 2021 | Software for Dataset-wide XAI: From Local Explanations to Global Insights with Zennit, CoRelAy, and ViRelAy;
Ali, A., Schnake, T., Eberle, O., Montavon, G., Müller, K.-R., & Wolf, L. | 2022 | XAI for Transformers: Better Explanations through Conservative Propagation;
Achtibat, R., et al. | 2024 | AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for Transformers;
Bakish, Y., Zimerman, I., Chefer, H., & Wolf, L. | 2025 | Revisiting LRP: Positional Attribution as the Missing Ingredient for Transformer Explainability;
Berend, J., Achtibat, R., Schäffer, D., Binder, A., Samek, W., Lapuschkin, S., & Dreyer, M. | 2026 | ResLRP: The Role of Residual Cancellation in Attribution Instability in Vision Transformers;
{%- endcapture -%}

{% include reference.html ref=references %}

<script src="{{ '/assets/js/posts/label-free-lrp.js' | relative_url }}" defer></script>
