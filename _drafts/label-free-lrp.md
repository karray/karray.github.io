---
layout: post
title: LRP for Modern Vision Models and the Rules It Was Missing
styles:
  - /assets/css/posts/label-free-lrp.css
tags: XAI, LRP, CNN, ViT, Self-Supervised Learning
description: How layer-wise relevance propagation explains a network and why it breaks on parts it has no rule for.
---

When a vision model makes a decision, we often want to know which pixels it relied on. Layer-wise relevance propagation (LRP) answers this by passing the output back through the network, dividing it among the inputs by how much each contributed. What counts as a contribution depends on the operation, so each operation needs a rule of its own. Without the right rules, the maps turn into noise, or look fine while carrying the wrong amounts. This post explains how LRP works and fills in the rules that modern models were missing.

<!--more-->

## What is relevance?

Think of the output of a network as an amount of water that we pour back through it. At every layer, the amount held by each unit is divided among the units that fed into it, and a unit that contributed more receives more. The amount that reaches each pixel tells us how much that pixel contributed. This amount is called relevance, and the procedure is layer-wise relevance propagation [<a href="#bach2015on" data-ref="bach2015on">Bach et al.</a>].

The figure below does this for a small network that reads handwritten digits. Its left half computes the prediction, and its right half passes the score back to the pixels. You can draw your own digit on the input.

{% include posts/label-free-lrp-water.html %}

How the relevance is divided at each layer is decided by a rule. Each kind of layer needs one that fits what it computes, and choosing them well is most of the work. The common rules and when to use them are described in *Layer-Wise Relevance Propagation: An Overview* [<a href="#montavon2019layer-wise" data-ref="montavon2019layer-wise">Montavon et al.</a>]. Since relevance is only divided and never created, the amounts at every step should add up to what we started with. This property is called conservation. In the figure, the number under each column is the total that arrives there, and it stays the same from column to column.

Usually, the relevance starts with the score of the class being explained. But here, we start with 1 instead, so that every column reads $1.00$ and conservation is easy to see. The maps stay the same. Each step only hands on fixed shares of what arrives, so every amount grows and shrinks with the start, and starting with 1 just turns each number into a share of the score. For a negative score, the figure starts with −1, so that the signs stay right.

## Relevance Is Not a Gradient

Most explanation methods for neural networks are built on the gradient. The gradient answers a different question than relevance, and the difference is the key to this post.

Take a single neuron with two inputs. It multiplies each input by a weight and adds the results. Say the inputs are $x_1 = 2$ and $x_2 = 1$, and the weights are $w_1 = 1.5$ and $w_2 = 1$. The neuron then outputs $1.5 \times 2 + 1 \times 1 = 3 + 1 = 4$.

The gradient asks how much the output changes when one input changes a little. Raising $x_1$ by a small step raises the output by $1.5$ such steps, and raising $x_2$ raises it by exactly one step. The gradients are therefore just the weights, $1.5$ and $1$. They tell us how sensitive the output is to each input, not how much each input contributed.

Relevance asks how much of the output came from each input. An input contributes its value times its weight, so $x_1$ contributed $3$ of the $4$ and $x_2$ contributed $1$. Unlike the gradient, this depends on the input itself. An input of zero contributes nothing and receives no relevance, however large its weight. LRP divides whatever relevance arrives at the neuron in these proportions. Every layer of a network is made of such neurons, so the same step, repeated layer by layer, carries the relevance from the output to the pixels.

LRP is computed with the same backward pass that computes gradients. Each rule replaces the gradient of its layer by a share of the relevance. A layer that has no rule does not raise an error. It simply passes the relevance back the way a gradient would. Our neuron would then pass back $1.5 \times 4 = 6$ to the first input and $1 \times 4 = 4$ to the second, 10 in total from the 4 that arrived.

The figure below shows the same difference in the digit network from above. Its backward half can pass the relevance back with LRP or the way a gradient would. Passed back like a gradient, the total changes from layer to layer and part of it turns negative, shown in blue. The pixel map also reaches into the black background, which contributed nothing to the score. With LRP, every column holds the same total, because the rules only divide.

The second switch decides what happens to negative amounts. The basic LRP rule, LRP-0, keeps them. An input that worked against the score then receives negative relevance, and the total still stays the same from column to column. The $\text{z}^+$ rule, which the first figure uses, follows only positive contributions, so all of the relevance stays positive. Its counterpart for gradients, guided backpropagation, drops the negative part of the gradient at every unit. Passing through the next weights can make part of it negative again, so a column still shows some blue before its units drop it. The pixels have no units that could drop anything, so some blue remains in the final map. The $\text{z}^+$ rule never creates negative amounts in the first place. With guided backpropagation, the total also still changes from layer to layer.

{% include posts/label-free-lrp-water.html gradient=true mode="gradient" kicker="Relevance or gradient" caption="The digit network from the first figure. The first switch decides how its right half passes the score back, with LRP or the way a gradient would. The second keeps the negative amounts or passes back only positive ones, which gives the gradient, guided backpropagation, LRP-0 and the $\text{z}^+$ rule. Red is positive and blue is negative. The number under each column is the total that arrives there. Draw a digit or hover another class to compare." %}

For a few parts, such as a ReLU, the gradient and the rule happen to agree. For the parts of modern encoders described below, they do not. The result can be a map that looks fine but carries the wrong amount, or pure noise, and the first attempt below shows both.

## What Is Inside a Modern Encoder

In the [previous post]({% post_url 2026-10-01-unsupervised-feature-attribution %}), we used LRP to explain self-supervised vision encoders without any labels. An encoder is the part of a vision model that turns an image into features, a grid of vectors that describes what is where in the image. The encoders in this post come in two families. ConvNeXt is a convolutional network, like the older ResNet, redesigned with ideas from transformers. Vision transformers (ViTs) cut the image into patches of sixteen by sixteen pixels and turn each patch into a vector of numbers, called a token. Swin sits in between. It is a transformer that, like a convolutional network, looks at the image in local windows and shrinks it in stages.

Both families are a stack of small networks, called blocks. The blocks do not simply hand their result to the next block. Instead, the features of the image flow along one main path through the whole network, called the residual stream. Each block reads from the stream, computes something, and adds its result back to it. The stream therefore keeps everything that earlier blocks wrote, and each block only adds a small correction. The addition that writes a block's result back is called a residual connection, and it is what lets very deep networks train.

Since every block adds to the stream, its values can grow large on the way through the network. Each block therefore brings what it read into a common range before most of its computation. This step is called LayerNorm. It subtracts the mean of each position's values and divides them by their spread.

The rest of a block is mostly layers with weights, convolutions and linear layers, which combine their inputs like the neuron above. Between them sits GELU, a smooth bend applied to every value on its own, a softer version of the ReLU in older networks. Transformer blocks add one more step, called attention. There, each token collects information from the other tokens by taking a weighted average of them, with weights computed from the tokens themselves.

[Figure to come. A ConvNeXt block and a ViT block on the residual stream, with each part labelled, and the parts that zennit covers marked differently from those it does not.]

Two libraries provide ready-made LRP rules for PyTorch. [zennit](https://github.com/chr5tphr/zennit) covers convolutional networks such as VGG and ResNet [<a href="#anders2021software" data-ref="anders2021software">Anders et al.</a>]. It never edits the model's code. It attaches its rules to the model's layers only while an explanation runs. Where a block needs a different form for a rule to apply, a canonizer rewrites the block for that time and restores it afterwards. [LXT](https://github.com/rachtibat/LRP-eXplains-Transformers), the implementation of AttnLRP, covers language models and the original vision transformer [<a href="#achtibat2024attnlrp" data-ref="achtibat2024attnlrp">Achtibat et al.</a>].

Both libraries have rules for the layers with weights, and these carry over to our encoders. zennit has no rules for LayerNorm or attention, and its canonizers for residual connections are written for ResNet. LXT does have rules for LayerNorm, GELU and attention, but it installs them by changing PyTorch's own classes for the whole program. The models we used come from the timm library, which implements LayerNorm and attention with its own code, so most of LXT's rules would silently not apply. We therefore wrote the missing rules in zennit's way, as canonizers that rewrite each kind of block while an explanation runs.

## Where the Relevance Starts

LRP was designed to explain a prediction, so it usually starts from a class score. A self-supervised encoder has no class score. We start from its features instead, the grid of vectors at the end of the network, and use only their positive part. In the previous post, we averaged the same positive features into a coarse map, and LRP traces that map back to the pixels.

Each feature starts with its own positive value, and as in the digit figures, we scale the start so that it adds up to 1. The amount that arrives at the pixels is then simply the share of the start that made it all the way through. Here the scaling has a second benefit. The features add up to different totals in different models and images, and once scaled, the numbers can be compared across all of them. The maps, again, stay the same.

<details markdown="1">
<summary>Why only the positive part of the features?</summary>

The features of most modern encoders are signed. If we started from the raw values, positive and negative values would largely cancel. In ConvNeXt, Swin and the plain ViTs, the sum of all feature values keeps only 1 to 5% of their total magnitude, and for several of these models it is negative for almost every image. ResNets do not have this problem, because their features pass through a ReLU and are never negative. Using the positive part avoids the cancellation and gives inactive features no relevance at all.

</details>

The figure below shows where we ended up, with all the rules described in this post. On the left is where the relevance starts, the feature grid at the end of the network. To its right is the same relevance after it has been passed back to ever earlier layers, until it arrives at the pixels.

{% include posts/label-free-lrp-stack.html %}

The cat receives less relevance than the dog. The starting amount is shared among all features in proportion to their size, and the dog's features are larger, so the dog gets the bigger share.

## The First Attempt

We started where the libraries stop. We gave the layers with weights their usual rules and left everything else without a rule. The first column of the figure below shows the result for three encoders. In all of them, the map is noise.

{% include figure-gallery.html id="lrp-rules" data=site.data.galleries.lrp_rules class="wide" %}

The number under each map is how much of the starting amount of 1 arrives at the pixels. In the first column, it ranges from almost nothing in ConvNeXt to more than a thousand in the DINOv1 ViT. Neither is plausible for an amount that should only be divided. The parts without a rule passed the relevance back like a gradient, and the next sections take them one at a time. The two middle columns each add a single rule, and the last column has all of them.

## Residual Connections

At a residual connection, a block's result is added to the residual stream. Two things meet in this addition. One is the stream as it was before the block, which passes the block by and is called the skip. The other is what the block wrote, called the update. Going backward, the relevance of the sum has to be shared between the skip and the update.

An addition is a neuron whose two weights are both 1, so its gradient is 1 for both inputs. Without a rule, the skip and the update therefore each receive the full relevance of the sum. The relevance is copied instead of divided.

To watch this happen, we gave the digit network one block of its own. The block reads the eight units, brings them into a common range with a LayerNorm, passes them through a layer with weights, and adds its result back to the same eight units. The figure below draws the addition and the LayerNorm as nodes of their own, and each has a switch for its rule.

[Figure to come. The digit network with one residual block on its eight units, drawn with its addition and its LayerNorm. One switch lets the addition split or copy, the other lets the LayerNorm pass relevance back with or without its rule. Each column shows its total.]

Set the addition to copy. The eight units below the block now receive relevance twice, once through the skip and once through the update, and their total grows to nearly twice what arrived. One block can at most double the amount. A ConvNeXt has 36 such additions in a row and a ViT has 24, and in a real encoder the copies pile up from block to block.

The third column of the gallery shows the result. There, LayerNorm already has its rule, but every addition copies. In the ViT, 450 thousand times the starting amount reaches the pixels. In ConvNeXt it is 720 times, and in Swin almost three trillion times. Each map is scaled to its own maximum, which hides the size of these numbers, and in the ViT the objects still look right.

Older networks are not spared. A ResNet-50 has 16 residual blocks. When its additions copy, it produces exactly the same map as when they divide the relevance equally, only 65,536 times larger. Because every map is scaled for display, the familiar ResNet maps never showed this.

The fix is a canonizer that rewrites every addition, so that LRP can apply a rule to it. The rule then divides the relevance between the skip and the update. How to divide it turned out to be a question of its own, and we return to it after the remaining parts.

## LayerNorm

LayerNorm subtracts the mean of a position's values and divides them by their spread. If we add the same number to every input, the mean grows by that number and the output does not change at all. As a consequence, the gradient of LayerNorm always sums to zero over its inputs.

Without a rule, relevance that passes through a LayerNorm is therefore split into positive and negative parts of exactly the same size. In the small block above, set the addition to split and the LayerNorm to no rule. The relevance that enters the LayerNorm leaves it as red and blue lines, positive and negative amounts that add up to zero. What the update path carried is lost as a whole, and the mix of signs travels on towards the pixels.

A real encoder has a LayerNorm in every block, so this happens dozens of times on the way to the pixels. The positive and negative amounts end up in neighbouring places, and the map turns into noise. This is why the second column of the gallery, where every addition already divides correctly, still shows noise.

Our rule holds the spread fixed during the backward pass, while the forward pass stays exactly the same. LayerNorm then behaves like a linear layer, and relevance passes through it with the usual rule for linear layers. Conservative propagation proposed this treatment for transformers [<a href="#ali2022xai" data-ref="ali2022xai">Ali et al.</a>], and AttnLRP uses an equivalent rule [<a href="#achtibat2024attnlrp" data-ref="achtibat2024attnlrp">Achtibat et al.</a>].

With this rule alone, in the third column, the objects appear in the ViT, but with the inflated amounts from the copying additions. In ConvNeXt and Swin, the maps stay noise. Only in the last column, which has both rules, do the objects appear in all three models with a plausible amount.

## Activation Functions

An activation function such as GELU changes each value on its own and does not mix values. The usual LRP treatment passes relevance through it unchanged, which AttnLRP calls the identity rule [<a href="#achtibat2024attnlrp" data-ref="achtibat2024attnlrp">Achtibat et al.</a>]. Without a rule, the relevance is multiplied by the slope of the function, as a gradient would be. For most inputs the slope of GELU is below one, so a little relevance leaks away at every activation.

zennit applies the identity rule to the activation functions it knows. The ConvNeXt from timm implements GELU with its own class, which zennit does not recognize, and nothing warns about it. With the rule in place, 0.66 of the starting amount reaches the pixels. Without it, only 0.48 arrives, so about a quarter is lost on the way. The two maps look the same. This was the first lesson of the project. A plausible map is no proof that every layer was handled.

[Figure to come, if needed.]

## Attention

In attention, each token takes a weighted average of what the other tokens send, and the weights are computed from the tokens themselves. Without a rule, relevance follows the gradient. Each source token then receives relevance in proportion to how strongly it is attended to, regardless of how much it actually sent. Another part flows into the computation of the weights, in positive and negative amounts that cancel.

We hold the weights fixed during the backward pass, so that relevance follows what each token actually sent. This is the approach of conservative propagation [<a href="#ali2022xai" data-ref="ali2022xai">Ali et al.</a>]. AttnLRP goes further and also passes relevance through the computation of the weights, with rules designed for it, which made its explanations of classifiers more faithful [<a href="#achtibat2024attnlrp" data-ref="achtibat2024attnlrp">Achtibat et al.</a>]. In our encoders, holding the weights fixed changed the maps only a little. Without it, Swin's maps gain a few bright squares, and the amount at the pixels changes by about a tenth.

Attention also explains a difference you can see in the stacked figure above. A convolution only combines neighbouring cells, so relevance in ConvNeXt stays close to where it started. Attention connects every token with every other one, so in the ViT, relevance can jump across the image within a single block.

[Figure to come, if needed.]

## Who Gets the Credit at a Residual Connection

Once every addition can receive a rule, LRP must decide how much of the relevance goes to the skip and how much to the update. There are several reasonable answers, and they behave very differently.

- An **equal** split gives each part half, regardless of the numbers.
- A **proportional** split gives each part what it contributed, like the neuron above. This is the standard LRP treatment of an addition, and it is what LXT and AttnLRP do implicitly. A part that worked against the sum receives negative relevance.
- A **magnitude** split divides by size and ignores the sign. We introduced it for MaxViT and the ViTs, after the proportional split made some of their maps unstable in our early experiments.
- A **γ split** works like the proportional one, but gives extra weight to the part whose sign agrees with the sum. It was proposed by Berend et al. while we were writing this post [<a href="#berend2026reslrp" data-ref="berend2026reslrp">Berend et al.</a>].

When the skip and the update have the same sign, the proportional, magnitude and γ splits agree. They disagree when the update works against the skip. The proportional split keeps the information that the update worked against the sum, but when the two nearly cancel, it divides by a sum close to zero and hands out huge positive and negative amounts. The magnitude split never grows, but it credits the opposing update as if it had helped. The γ split keeps the sign and stays bounded. Berend et al. identify such cancellation inside transformer blocks as a main reason why LRP maps of ViTs become noisy.

{% include figure-gallery.html id="lrp-residual" data=site.data.galleries.lrp_residual class="wide" %}

In the maps, the equal split stands out first. In a deep network, it sends half of the relevance into every block's update, where part of it is absorbed. Very little reaches the pixels, as little as 0.04 for the DINOv1 ViT. The proportional split delivers the most relevance, but in DINOv1 it also spreads it over the grass and the trees. The magnitude and γ splits keep more of it on the animals. The γ split passes on more relevance than the magnitude split and is a little more detailed.

We chose our rules by looking at maps like these and by checking that small changes to the input did not change the maps much. We did not test which rule is more faithful to the model, and on a handful of images we would not want to claim one. For anyone who wants to choose between them, Berend et al. compare the γ split with the standard rules on several faithfulness measures.

<details markdown="1">
<summary>A few numbers behind the maps</summary>

On eight ImageNet images per model, we measured three things for each rule. The first is how much relevance reaches the pixels. The second is how much the map changes when the input moves by a tiny amount. The third is whether the map follows its start, which we tested by starting only from the feature cells inside the object box and then only from those outside.

The equal split delivered the least relevance (0.00 to 0.12 for most models) and its maps barely depended on where they started. The proportional split delivered the most (up to 0.97) and followed its start most closely, and it stayed stable on these images. The magnitude and γ splits were the most stable. The γ split followed its start almost as closely as the proportional one. The maps of the supervised ViT barely depended on their start under any rule.

</details>

## Smaller Pieces and the Rule Inside the Blocks

A few smaller parts needed a rule as well. Some blocks multiply their update by a learned constant, called layer scale, and MaxViT scales its channels with learned gates. Both simply pass relevance on. Batch normalization, used in older networks, can be folded into the convolution before it, which turns two layers into one and leaves the forward pass unchanged. Folding is only exact when the convolution feeds nothing else, so we check that before merging.

{%- comment -%}TODO: cite canonization / batch-norm folding work (e.g. Motzkus et al. 2022; Hui and Binder 2019) after verifying.{%- endcomment %}

The layers with weights inside the blocks had a rule from the start, but we still had to pick one. zennit offers several, and they trade detail against focus. We use the Gamma rule, which lies between LRP-0 and the $\text{z}^+$ rule. Like LRP-0, it keeps negative contributions, but it gives the positive ones extra weight, and the larger its parameter γ, the closer it comes to $\text{z}^+$. This makes the maps less noisy. AttnLRP recommends it for the linear layers of vision transformers [<a href="#achtibat2024attnlrp" data-ref="achtibat2024attnlrp">Achtibat et al.</a>].

<details markdown="1">
<summary>All rules at a glance</summary>

<div class="lrp-scroll" markdown="1">

| | ResNets | ConvNeXt | Swin | MaxViT | Plain ViTs |
|---|---|---|---|---|---|
| first layer | $\text{z}^+$ | WSquare | WSquare | WSquare | $\text{z}^+$ |
| layers inside blocks | $\text{z}^+$ | Gamma | Gamma | Gamma | Gamma |
| activations, layer scale, gates | pass | pass | pass | pass | pass |
| LayerNorm | — | spread held fixed | spread held fixed | spread held fixed | spread held fixed |
| attention | — | — | weights held fixed | weights held fixed | weights held fixed |
| residual split | equal | proportional | proportional | magnitude | magnitude |

</div>

</details>

## The Last Step to the Pixels

The first layer of many modern encoders cuts the image into patches and turns each patch into a feature vector, four by four pixels in ConvNeXt and Swin and sixteen by sixteen in a ViT. Its rule decides how a patch's relevance is spread over its pixels, and with that, most of the fine detail in the map. zennit offers rules for this step, but which one fits a patch layer was ours to find out.

- **$\text{z}^+$** follows the positive contributions of the actual pixel values, as in the digit network.
- **WSquare** spreads it according to the squared weights alone and ignores the pixel values. It is recommended for inputs without natural bounds [<a href="#montavon2019layer-wise" data-ref="montavon2019layer-wise">Montavon et al.</a>].
- **Gamma** is the rule we use inside the blocks.

Because WSquare ignores the pixels, it fills every patch with the same pattern, scaled by the patch's relevance. In the DINOv2 and DINOv3 ViTs, this one repeated pattern accounts for more than 99% of the detail within the patches. $\text{z}^+$ follows the image inside each patch instead, which is why we use it for the ViTs.

{% include figure-gallery.html id="lrp-stems" data=site.data.galleries.lrp_stems_rules class="wide" %}

Look closely at the maps, and the borders between the patches become visible. WSquare paints visible edges along them. $\text{z}^+$ shows fewer edges, and Gamma almost none. The number under each map measures this as the contrast between neighbouring pixels across a patch border and within a patch, where 1 means no edges. Over the five images, the median for ConvNeXt is 2.11 with WSquare, 1.66 with $\text{z}^+$ and 0.97 with Gamma. Swin and the DINOv1 ViT behave the same way. The rules in our table are the ones we settled on early, WSquare for ConvNeXt, Swin and MaxViT and $\text{z}^+$ for the ViTs. These measurements suggest that Gamma deserves a closer look.

## Where the Relevance Goes

With all rules in place, not all of the starting amount reaches the pixels, and how much arrives differs a lot between models.

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

{%- comment -%}Median over 8 ImageNet images, tutorial rules, start = 1. Source: lrp/temp/lrp_post/NOTES.md.{%- endcomment %}

The rest stays inside the network. Most layers add a constant, the bias, to their output. The bias is not an input, yet part of the output is due to it, so the rules leave part of the relevance with it. Small stabilizing terms in the rules keep a little more. In transformers, the class token and the register tokens take their share as well, because they are learned parameters rather than parts of the image.

Is this a flaw? AttnLRP argues that it is not. Following the original LRP paper, it treats a bias as an extra input whose value is always one, so that its relevance is "preserved, rather than lost" [<a href="#achtibat2024attnlrp" data-ref="achtibat2024attnlrp">Achtibat et al.</a>]. One could force all relevance onto the pixels, by spreading the bias's share over the inputs or by leaving the bias out of the rules. AttnLRP shows that both options can make the results numerically unstable or flip their signs. In our own tests, leaving out the bias kept all of the relevance in a ResNet but produced invalid numbers in ConvNeXt. We accept that part of the relevance stays inside the network and report how much reaches the pixels.

Position embeddings are the one place where our rules currently err in the other direction. A ViT adds them to the patch tokens with a plain addition, so the patches receive the share that belongs to the positions. Bakish et al. show that position embeddings carry relevance of their own and that ignoring them violates conservation [<a href="#bakish2025revisiting" data-ref="bakish2025revisiting">Bakish et al.</a>].

## What We Checked and What We Did Not

Since a broken rule can still produce a plausible map, we relied on a few simple checks rather than on the pictures. They are worth running for any model one adds.

- The model's output must not change when the rules are installed. Every rewrite only changes the backward pass.
- Every layer must receive a rule. Counting the layers that have no rule catches cases like the GELU above.
- The amount reaching the pixels should be plausible. It should not exceed the starting amount by orders of magnitude.
- After the explanation, the model must be restored exactly, so that later predictions are not affected.

We run LRP on the CPU. It takes a few seconds per image and gives identical numbers on every machine.

{%- comment -%}TODO: these checks come from the research code; add them to notebook 02 (and xai_utils) so the claim holds for the published package.{%- endcomment %}

Several questions remain open.

- The maps depend on the rules. The residual split alone changes how much relevance arrives and where it goes.
- In plain ViTs, the maps barely depend on where the relevance starts. For the supervised ViT, this held under every residual rule we tried. If the start hardly matters, it is not obvious what exactly such a map explains.
- We did not measure faithfulness. Readers who need it will find established measures in the work of Achtibat et al. and Berend et al.
- MAE keeps a strong grid in its maps under every first-layer rule we tried, and none of our other changes removed it either. Part of it seems to come from MAE's own learned weights, which favor the borders of each patch. Where exactly it comes from is a question for a separate post.

## Use It Yourself

All rules live in a small package that comes with the [companion notebook](https://github.com/karray/deep-unboxing) {%- comment -%}TODO: link to 02_... once it exists{%- endcomment -%}. Explaining an encoder takes a few lines.

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
barely depend on their start; which residual split to trust; where the grids come from (next post).{%- endcomment %}

Modern vision encoders are built from parts that the LRP libraries do not cover. Where a part has no rule, LRP does not fail. It quietly falls back to the gradient. Additions then copy the relevance, LayerNorm splits it into amounts that cancel, and GELU lets some of it leak away. Sometimes the map shows the damage, as with the noise in the first attempt. Sometimes it does not, as with the copying additions in the ViT or the missing GELU rule in ConvNeXt. Checking the amounts, and not only the pictures, is what caught these cases.

The rules described here make the maps of ConvNeXt, Swin and ViTs plausible again, but plausible is not the same as faithful. Two questions from the figures stay with us. Why do the maps of plain ViTs barely depend on where the relevance starts? And which of the residual splits describes what the model actually does?

## References

{%- capture references -%}
Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K.-R., & Samek, W. | 2015 | On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation;
Montavon, G., Binder, A., Lapuschkin, S., Samek, W., & Müller, K.-R. | 2019 | Layer-Wise Relevance Propagation: An Overview;
Anders, C. J., Neumann, D., Samek, W., Müller, K.-R., & Lapuschkin, S. | 2021 | Software for Dataset-wide XAI: From Local Explanations to Global Insights with Zennit, CoRelAy, and ViRelAy;
Ali, A., Schnake, T., Eberle, O., Montavon, G., Müller, K.-R., & Wolf, L. | 2022 | XAI for Transformers: Better Explanations through Conservative Propagation;
Achtibat, R., Dreyer, M., Eisenbraun, I., Bosse, S., Wiegand, T., Samek, W., & Lapuschkin, S. | 2023 | From Attribution Maps to Human-Understandable Explanations through Concept Relevance Propagation;
Achtibat, R., et al. | 2024 | AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for Transformers;
Bakish, Y., Zimerman, I., Chefer, H., & Wolf, L. | 2025 | Revisiting LRP: Positional Attribution as the Missing Ingredient for Transformer Explainability;
Berend, J., Achtibat, R., Schäffer, D., Binder, A., Samek, W., Lapuschkin, S., & Dreyer, M. | 2026 | ResLRP: The Role of Residual Cancellation in Attribution Instability in Vision Transformers;
{%- endcapture -%}

{% include reference.html ref=references %}

<script src="{{ '/assets/js/posts/label-free-lrp.js' | relative_url }}" defer></script>
