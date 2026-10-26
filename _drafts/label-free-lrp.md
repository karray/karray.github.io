---
layout: post
title: Why LRP Fails Quietly on Modern Vision Models
styles:
  - /assets/css/posts/label-free-lrp.css
tags: XAI, LRP, CNN, ViT, Self-Supervised Learning
description: Where LRP has no rule for a part of a network, it falls back to the gradient without a warning, and the maps turn into noise or look right while the numbers behind them are wrong.
---


Image classifiers do not always look where we expect. One well-known model recognised horses by a copyright tag in the corner of the photos, and only a map of the pixels it relied on revealed the shortcut. Layer-wise relevance propagation (LRP) is one way to draw such maps, but on modern vision models it can quietly go wrong, producing noise or maps that look fine while carrying the wrong amounts. This post shows why and fills in the missing rules, which come with a [companion notebook](https://github.com/karray/deep-unboxing).

<!--more-->

## What is relevance?

Think of the output of a network as water that we pour back through it. At every layer, the water held by each unit is divided among the units that fed into it, and a unit that contributed more receives more. Whatever reaches a pixel tells us how much that pixel contributed. This quantity is called relevance, and the procedure is layer-wise relevance propagation [<a href="#bach2015on" data-ref="bach2015on">Bach et al.</a>]. The horse shortcut was found with such maps [<a href="#lapuschkin2019unmasking" data-ref="lapuschkin2019unmasking">Lapuschkin et al.</a>].

The figure below does this for a small network that reads handwritten digits. Its left half computes the prediction, and its right half passes the score back to the pixels. You can draw your own digit on the input.

{% include posts/label-free-lrp-water.html %}

It helps to think of each layer as a small network of its own. It receives inputs, computes an output and hands it on. During the prediction, the network keeps what each of these small networks received. On the way back, each one divides the relevance that arrives at its output among its own inputs, without looking at the rest of the network. In the figure, the left half is this stored prediction, and the right half reuses it step by step.

How a layer divides the relevance is decided by a rule, and each kind of layer needs one that fits what it computes. Choosing these rules well is most of the work, and *Layer-Wise Relevance Propagation: An Overview* describes the common ones and when to use them [<a href="#montavon2019layer-wise" data-ref="montavon2019layer-wise">Montavon et al.</a>]. Since relevance is only divided and never created, the total at every step should equal what we started with. This property is called conservation, and in the figure the number under each column stays the same from column to column.

The relevance usually starts with the score of the class being explained. Here it starts with 1 instead, so that every column reads $1.00$ and conservation is easy to see. The maps do not change, because each step hands on fixed shares of what arrives, and starting with $1$ simply turns every number into a share of the score. For a class with a negative score, the figure starts with 0 and shows an empty map. The figure uses the $\text{z}^+$ rule, which follows only the contributions that raised the score, so it cannot say what pulled the score below zero.

## Relevance is not a gradient

Most explanation methods for neural networks are built on the gradient, but the gradient answers a different question than relevance. A single neuron is enough to see the difference. Take one with inputs $x_1 = 2$ and $x_2 = 1$ and weights $w_1 = 1.5$ and $w_2 = -1$. It multiplies each input by its weight and adds the results, so it outputs $3 - 1 = 2$. The gradient asks how much the output changes when an input changes a little, and for this neuron the answer is simply the weights, $1.5$ and $-1$.

The figure below shows this neuron. Like the digit network, it is mirrored around the output. Drag the inputs and the weights on the forward side, and the relevance flows back to the inputs on the other side, red for positive and blue for negative.

<figure class="lrp-fig lrp-neuron">
  <div data-lrp-toy="neuron"><p class="lrp-small">This figure needs JavaScript.</p></div>
</figure>

Relevance asks instead how much of the output came from each input. An input contributes its value times its weight, so $x_1$ contributed $3$ and $x_2$ contributed $-1$, while an input of zero would contribute nothing, however large its weight. LRP divides the relevance that arrives in these proportions, and this basic rule is called LRP-0. It passes $3$ back to $x_1$ and $-1$ to $x_2$, so the input that worked against the output receives negative relevance, and the total stays $2$. The $\text{z}^+$ rule of the first figure follows only the positive contributions, as its name says, with $z$ for a contribution and $+$ for its positive part. It gives all of the $2$ to $x_1$ and nothing to $x_2$, so its relevance never turns negative. Since every layer is made of such neurons, the same step, repeated layer by layer, carries the relevance to the pixels.

LRP runs on the same backward pass that computes gradients, and each rule replaces the gradient of its layer with a share of the relevance. A layer without a rule raises no error and simply passes the relevance back the way a gradient would, here $1.5 \times 2 = 3$ and $-1 \times 2 = -2$, which makes 1 from the 2 that arrived. We will call this the gradient fallback, and it is behind every failure in this post. The switch in the figure above shows it for any inputs and weights.

The next figure runs the whole digit network both ways. With the gradient, the total changes from layer to layer, part of the relevance turns negative, shown in blue, and the pixel map reaches into the background, which contributed nothing to the score. With LRP, every column holds the same total, because the rules only divide.

{% include posts/label-free-lrp-water.html gradient=true mode="gradient" kicker="Relevance or gradient" caption="The digit network from the first figure. The first switch decides how its right half passes the score back, with LRP or the way a gradient would. The second keeps the negative contributions or passes back only positive ones, which gives the gradient, guided backpropagation, LRP-0 and the $\text{z}^+$ rule. Red is positive and blue is negative. The number under each column is the total that arrives there. Draw a digit or hover another class to compare." %}

The second switch decides what happens to negative contributions, which gives four methods in total. The plain gradient is ordinary backpropagation, the same backward pass that trains the network, and its map at the pixels is known as a saliency map [<a href="#simonyan2014deep" data-ref="simonyan2014deep">Simonyan et al.</a>]. LRP-0 and the $\text{z}^+$ rule are the two rules from our neuron. Guided backpropagation is the gradient's counterpart to $\text{z}^+$ and passes back only the positive part of the gradient, yet some blue remains and the totals still change.

<details markdown="1">
<summary>What is guided backpropagation?</summary>

Guided backpropagation was introduced to produce sharper pictures of what a convolutional network responds to [<a href="#springenberg2015striving" data-ref="springenberg2015striving">Springenberg et al.</a>]. It runs ordinary backpropagation, but at every unit it lets only the positive part of the gradient through, and only if the unit was active in the prediction. It drops the negative part only at the units, though. Passing through the next weights can make part of the gradient negative again, so a column still shows some blue before its units drop it. The pixels have no units that could drop anything, so some blue remains in the final map. Since nothing is divided, the totals also still change from column to column. The $\text{z}^+$ rule never creates negative relevance in the first place.
</details>

The two rules also treat the classes differently. In the figure above, LRP-0 starts with −1 for a class with a negative score, and the maps of different classes clearly differ. The $\text{z}^+$ rule starts with nothing for such a class, as in the first figure. For a class with a positive score, it draws almost the same map as for the prediction, because the positive contributions it follows depend on the input and hardly on the class. You can hover the classes in both modes to compare.


For a few parts, such as the ReLUs of the digit network, the gradient and the rule happen to agree, and the difference between the gradient and LRP-0 in the figure comes from the weighted layers alone. Most of the parts that make modern encoders modern are not among them.

## The first attempt

The [previous post]({% post_url 2026-10-01-unsupervised-feature-attribution %}) applied LRP to self-supervised vision encoders, which learn from images without any labels. Instead of a score for each class, such an encoder turns an image into an embedding, a vector of numbers that describes the whole image. Inside, it computes a feature vector for each cell of a coarse grid over the image, and the embedding it outputs is simply their average. The relevance can therefore start from the features directly. Only their positive part is used, scaled to add up to 1 as in the digit figures.

<details markdown="1">
<summary>What are features, and how does the relevance start from them?</summary>

What a feature vector stands for depends on the architecture. A vision transformer (ViT) cuts the image into patches of sixteen by sixteen pixels and turns each patch into a feature vector, called a token, so its grid has $14 \times 14$ cells, and attention lets every token draw on the whole image. Convolutional networks call their feature vectors activations, and each one depends on a region of the image called its receptive field, which grows with every layer. ConvNeXt, Swin and MaxViT shrink the image in stages to a grid of $7 \times 7$ cells. Each feature vector holds several hundred numbers, called channels, for example $768$ in ViT-B and $1024$ in ConvNeXt Base.

Each positive number in the grid, one for each cell and channel, starts with its value divided by the sum of the positive numbers over every cell and channel, and negative numbers start with $0$. The starting relevance then adds up to $1$, and whatever arrives at the pixels is the share that made it all the way through. Since the features add up to different totals in different models and images, this scaling also makes the numbers comparable across all of them. The $\text{z}^+$ rule gives the same result when each channel of the embedding starts with its share of this sum and passes it back through the averaging.

Only the positive part is used because the features of most modern encoders are signed, so if we started from the raw values, positive and negative values would largely cancel. In ConvNeXt, Swin and the plain ViTs, the sum of all feature values keeps only 1 to 5% of their total magnitude, and for several of these models it is negative for almost every image. ResNets do not have this problem, because their features pass through a ReLU and are never negative. Using the positive part avoids the cancellation and gives inactive features no relevance at all.

</details>

The encoders here include ConvNeXt, a modern convolutional network, plain vision transformers (ViTs), and Swin and MaxViT, which combine ideas from both. All of them compute most of their output with convolutions and linear layers, which multiply each input by a weight and add the results. We will call them weighted layers. Two PyTorch libraries provide ready-made LRP rules for them and for more. [zennit](https://github.com/chr5tphr/zennit) covers convolutional networks such as VGG and ResNet [<a href="#anders2021software" data-ref="anders2021software">Anders et al.</a>]. Instead of editing the model's code, it attaches its rules to the layers while an explanation runs, and where a block needs a different form for a rule to apply, a canonizer rewrites the block for that time and restores it afterwards. [LXT](https://github.com/rachtibat/LRP-eXplains-Transformers), the implementation of AttnLRP, covers language models and the original vision transformer [<a href="#achtibat2024attnlrp" data-ref="achtibat2024attnlrp">Achtibat et al.</a>].

<details markdown="1">
<summary>Why not simply use zennit or LXT?</summary>

Both libraries have rules for the weighted layers, and these carry over to the encoders here. zennit, however, has no rules for LayerNorm or attention, and its canonizers for residual connections are written for ResNet. LXT does have rules for LayerNorm, GELU and attention, but it installs them by changing PyTorch's own classes for the whole program. The models here come from the timm library, which implements LayerNorm and attention with its own code, so most of LXT's rules would silently not apply. The missing rules in this post are therefore implemented in zennit's way, as canonizers that rewrite each kind of block while an explanation runs.

</details>

So we start where the libraries stop, with the usual rules for the weighted layers and no rule for anything else. The first column of the figure below shows the result for three encoders, and in all of them the map is noise.

{% include figure-gallery.html id="lrp-rules" data=site.data.galleries.lrp_rules class="wide" %}

The number under each map is how much of the starting relevance of 1 arrives at the pixels. In the first column, it ranges from almost nothing in ConvNeXt to more than a thousand in the DINOv1 ViT, and neither is possible for something that should only be divided. Every part without a rule fell back to the gradient. The two middle columns each add a single rule, and the last column has all of them. Even there, less than 1 arrives, but for a different reason that has nothing to do with missing rules.

## The missing rules

Besides the weighted layers, each block contains a few other parts (such as residual connections, LayerNorm, GELU, layer scale and attention), and the libraries provide no rule for them that applies here. To see what each missing rule does, the small ConvNeXt below has the same parts as the real encoders, and its switches choose, for each part, between its rule and the gradient fallback.

{% include posts/label-free-lrp-cnx.html %}

### Residual connections

In a plain network, each layer passes on only its output. In these encoders, a skip carries the input of a block around it and adds it to the block's output, which we call the update, and together they form a residual connection. Each block thus computes only a change to its input. The running sum that passes from block to block carries the input together with every change added to it and is called the residual stream.

The addition at the end of each block works like a neuron with two weights of 1, so LRP gives the skip and the update the share each contributed. The gradient of an addition, however, is 1 for both inputs, so the fallback hands the full relevance to each of them, and the total doubles at every block of the small ConvNeXt.

A ConvNeXt has 36 residual connections in a row and a ViT has 24, and the copies pile up from block to block. In the third column of the gallery in the first attempt, LayerNorm already has its rule but every residual connection copies. As a result, 450 thousand times the starting relevance reaches the pixels in the ViT, 720 times in ConvNeXt and almost three trillion times in Swin. Because each map is scaled to its own maximum, none of this shows, and in the ViT the objects even look right.

The fix is a canonizer that rewrites every addition so that LRP can apply a rule to it.

### LayerNorm

Because every block adds to the residual stream, the values can grow large on the way through the network. Each block therefore brings its values into a common range with LayerNorm, which subtracts the mean of each position's values and divides them by their spread.

Adding the same number to every input of a LayerNorm raises their mean by that number and leaves the output unchanged, so the gradient of LayerNorm always sums to zero over its inputs. The fallback therefore splits whatever relevance passes through into positive and negative parts of exactly the same size, as the LayerNorm card under the small ConvNeXt shows.

A real encoder has a LayerNorm in every block, so this happens dozens of times on the way to the pixels. Which channels receive the negative parts is decided by the normalisation more than by the objects in the image. Once the relevance reaches the input, the two signs land on neighbouring pixels, and the map turns into noise. The second column of that gallery shows this, although every residual connection there already divides the relevance. Even the totals mean nothing, from 0.01 in ConvNeXt to millions in Swin.

The rule treats the spread as a constant on the way back, while the forward pass stays exactly the same. LayerNorm then behaves like a linear layer, and relevance passes through it with the usual rule for linear layers. Conservative propagation proposed this treatment for transformers [<a href="#ali2022xai" data-ref="ali2022xai">Ali et al.</a>], and AttnLRP uses an equivalent rule [<a href="#achtibat2024attnlrp" data-ref="achtibat2024attnlrp">Achtibat et al.</a>]. With this rule alone, in the third column, the objects appear in the ViT, though with the inflated totals of the copying residual connections, while ConvNeXt and Swin stay noisy. Only the last column, with all rules, shows the objects in all three models with a plausible total.

### GELU and layer scale

Two weighted layers in a row would act like a single one, so an activation function sits between them and bends each value on its own, which lets the network compute more than weighted sums. ResNet uses the ReLU, which sets negative values to zero, while transformers use GELU, a smooth version of the ReLU that lets small negative values through, and ConvNeXt adopted it from them. Layer scale multiplies each channel of a block's output by a learned number before it joins the residual stream.

GELU and layer scale change each value on its own and mix nothing, so LRP passes relevance through them unchanged, which AttnLRP calls the identity rule [<a href="#achtibat2024attnlrp" data-ref="achtibat2024attnlrp">Achtibat et al.</a>]. The gradient fallback instead multiplies the relevance by the slope of GELU or by the learned numbers of layer scale. For the small negative values that GELU lets through, the slope is close to zero, so the relevance they carry is lost. Layer scale shrinks some channels, grows others and flips the sign of a few.

zennit applies the identity rule to the activation functions it knows, but the ConvNeXt from timm implements GELU with its own class, which zennit does not recognize, and nothing warns about it. With the rule, 0.66 of the starting relevance reaches the pixels, and without it only 0.48, so about a quarter is lost on the way. Yet the two maps look the same. This was the first lesson of the project, that a plausible map is no proof that every layer was handled.

### Attention

Transformer blocks use attention in place of the convolution. A transformer represents the image as tokens, one feature vector per patch, and attention lets each token take a weighted average of what the other tokens send, with weights computed anew for every image. This is the one part that the small ConvNeXt lacks.

Under the gradient fallback, each token receives relevance in proportion to its attention weight alone, however much or little it sent. Another part flows into the computation of the weights, in positive and negative amounts that cancel. Because the weights of a token add up to 1, the gradient still divides the relevance among the tokens, only by weight and not by contribution. A missing attention rule therefore does far less harm than a missing rule for LayerNorm or the residual connections.

We can treat the weights as constants on the way back, as if they had been given and not computed from the tokens. No relevance then flows into their computation, and the weighted average works like a weighted layer, where each token receives relevance in proportion to its contribution, its weight times what it sent. This is the approach of conservative propagation [<a href="#ali2022xai" data-ref="ali2022xai">Ali et al.</a>]. AttnLRP goes further and also passes relevance through the computation of the weights, with rules designed for it, which made its explanations of classifiers more faithful [<a href="#achtibat2024attnlrp" data-ref="achtibat2024attnlrp">Achtibat et al.</a>]. In these encoders, treating the weights as constants changed the maps only a little. Without it, Swin's maps gain a few bright squares, and the total at the pixels changes by about a tenth.

### Smaller pieces

A few smaller parts needed a rule as well. MaxViT scales its channels with learned gates, which pass relevance on unchanged like layer scale. Batch normalization, used in older networks, can be folded into the convolution before it, which turns two layers into one without changing the forward pass. Folding is only exact when the convolution feeds nothing else, so the code checks that before merging.

{%- comment -%}TODO: cite canonization / batch-norm folding work (e.g. Motzkus et al. 2022; Hui and Binder 2019) after verifying.{%- endcomment %}

## Where the relevance goes

With a rule for every part, the maps make sense again, and we can follow the relevance on its way through the network. The figure below does this for four encoders. On the left is where the relevance starts, the features of the last layer, and to its right is the same relevance after it has been passed back to ever earlier layers, until it arrives at the pixels.

{% include posts/label-free-lrp-stack.html %}

The encoders differ in how far the relevance moves on its way. A convolution only combines neighbouring cells, so in ConvNeXt the relevance stays close to where it started. Attention connects every token with every other one, so in a ViT the relevance can jump across the image within a single block.

The bands between the layers also narrow on the way, because not all of the starting relevance reaches the pixels. How much arrives differs a lot between models, from 0.02 in ResNet-50 to 0.85 in the supervised ViT.

<details markdown="1">
<summary>Relevance at the pixels for each model</summary>

The numbers use the rules from the next section and are medians over eight ImageNet images.

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

</details>

{%- comment -%}Median over 8 ImageNet images, tutorial rules, start = 1. Source: lrp/temp/lrp_post/NOTES.md.{%- endcomment %}

The rest stays inside the network. Most layers add a constant, the bias, to their output. The bias is not an input, yet part of the output is due to it, so the rules leave part of the relevance with it, and small stabilizing terms in the rules keep a little more. In transformers, the class token and the register tokens, extra tokens that the model learns and adds to the patch tokens, take their share as well.

Is this a flaw? AttnLRP argues that it is not. Following the original LRP paper, it treats a bias as an extra input whose value is always one, so that its relevance is "preserved, rather than lost" [<a href="#achtibat2024attnlrp" data-ref="achtibat2024attnlrp">Achtibat et al.</a>]. One could force all relevance onto the pixels, either by spreading the bias's share over the inputs or by leaving the bias out of the rules, but AttnLRP shows that both can make the results numerically unstable or flip their signs. Leaving out the bias kept all of the relevance in a ResNet but produced invalid numbers in ConvNeXt. This post therefore leaves part of the relevance inside the network and reports how much reaches the pixels.

Position embeddings are the one place where these rules err in the other direction and send too much to the pixels. A ViT adds them to the patch tokens with a plain addition, so the patches receive the share that belongs to the positions. Bakish et al. show that position embeddings carry relevance of their own and that ignoring them violates conservation [<a href="#bakish2025revisiting" data-ref="bakish2025revisiting">Bakish et al.</a>].

## Choices that remain

Once no part falls back to the gradient, three choices are left. They decide how the relevance is split at each residual connection, which rule the weighted layers inside the blocks use, and how the first layer spreads relevance over the pixels. None of them breaks the maps, but each one changes them.

### Who gets the credit at a residual connection

A residual connection with a rule must still decide how much of the relevance goes to the skip and how much to the update. There are several reasonable answers, and they behave very differently.

- An **equal** split gives each part half, regardless of the numbers.
- A **proportional** split gives each part the share it contributed. This is the standard LRP treatment of an addition, and it is what LXT and AttnLRP do implicitly. A part that worked against the sum receives negative relevance.
- A **magnitude** split divides by size and ignores the sign. The code uses it for MaxViT and the ViTs, because the proportional split made some of their maps unstable in early experiments.
- A **γ split** works like the proportional one, but gives extra weight to the part whose sign agrees with the sum [<a href="#berend2026reslrp" data-ref="berend2026reslrp">Berend et al.</a>].

The proportional, magnitude and γ splits agree as long as the skip and the update have the same sign. They part ways when the update works against the skip. The proportional split then gives the update a negative share, which records that it pulled the sum down. But when the two almost cancel, this split divides by a sum close to zero and hands out huge amounts of both signs. The magnitude split ignores the sign. It never grows, but it credits the opposing update as if it had helped. The γ split keeps the sign and still stays bounded. Berend et al. see this kind of cancellation inside transformer blocks as a main reason why LRP maps of ViTs turn noisy.

{% include figure-gallery.html id="lrp-residual" data=site.data.galleries.lrp_residual class="wide" %}

In the maps, the equal split stands out first. In a deep network, it sends half of the relevance into every block's update, where part of it stays behind with the biases, so very little reaches the pixels, as little as 0.04 for the DINOv1 ViT. The proportional split delivers the most relevance, but in DINOv1 it also spreads it over the grass and the trees, while the magnitude and γ splits keep more of it on the animals. Of these two, the γ split passes on more relevance and is a little more detailed.

We can choose a split by looking at maps like these and by checking that small changes to the input do not change the maps much. Which split is more faithful to the model remains untested here, and a handful of images would not be enough to claim one. For anyone who wants to choose between them, Berend et al. compare the γ split with the standard rules on several faithfulness measures, which test whether the pixels a map highlights are the ones the model actually depends on.

<details markdown="1">
<summary>A few numbers behind the maps</summary>

The numbers come from eight ImageNet images per model and measure three things for each rule. The first is how much relevance reaches the pixels. The second is how much the map changes when the input moves by a tiny amount. The third is whether the map follows where the relevance starts, checked by starting only from the features inside the object box and then only from those outside.

The equal split delivered the least relevance (0.00 to 0.12 for most models), and its maps barely depended on where they started. The proportional split delivered the most (up to 0.97), followed its starting features most closely and stayed stable on these images. The magnitude and γ splits were the most stable, and the γ split followed its starting features almost as closely as the proportional one. The maps of the supervised ViT barely depended on their starting features under any rule.

</details>

### The rule inside the blocks

The weighted layers inside the blocks had a rule from the start, but zennit offers several, and they trade detail against focus. The encoders here use the Gamma rule, which lies between LRP-0 and the $\text{z}^+$ rule. Like LRP-0, it keeps negative contributions, but it gives the positive ones extra weight, and the larger its parameter γ, the closer it comes to $\text{z}^+$. This makes the maps less noisy, and AttnLRP recommends it for the linear layers of vision transformers [<a href="#achtibat2024attnlrp" data-ref="achtibat2024attnlrp">Achtibat et al.</a>].

### The last step to the pixels

The first layer of many modern encoders cuts the image into patches and turns each patch into a feature vector, four by four pixels in ConvNeXt and Swin and sixteen by sixteen in a ViT. Its rule decides how a patch's relevance is spread over its pixels, and with that, most of the fine detail in the map. zennit offers several rules for this step, but it is not obvious which one suits a patch layer.

- **$\text{z}^+$** follows the positive contributions of the actual pixel values, as in the digit network.
- **WSquare** spreads the relevance according to the squared weights alone and ignores the pixel values. It is recommended for inputs without natural bounds [<a href="#montavon2019layer-wise" data-ref="montavon2019layer-wise">Montavon et al.</a>].
- **Gamma** is the rule used inside the blocks.

Because WSquare ignores the pixels, it fills every patch with the same pattern, scaled by the patch's relevance. In the DINOv2 and DINOv3 ViTs, this one repeated pattern accounts for more than 99% of the detail within the patches. $\text{z}^+$ follows the image inside each patch instead, which is why it is the first-layer rule for the ViTs.

{% include figure-gallery.html id="lrp-stems" data=site.data.galleries.lrp_stems_rules class="wide" %}

Look closely, and the borders between the patches show up in the maps. WSquare paints clear edges along them, $\text{z}^+$ fewer, and Gamma almost none. The number under each map measures this as the contrast between neighbouring pixels across a patch border compared with inside a patch, where 1 means no edges. Over the five images, the median for ConvNeXt is 2.11 with WSquare, 1.66 with $\text{z}^+$ and 0.97 with Gamma, and Swin and the DINOv1 ViT behave the same way. The table below lists the rules chosen early on, WSquare for ConvNeXt, Swin and MaxViT and $\text{z}^+$ for the ViTs. By these numbers, Gamma would be the better choice for all of them.

<details markdown="1">
<summary>All rules at a glance</summary>

<div class="lrp-scroll" markdown="1">

| | ResNets | ConvNeXt | Swin | MaxViT | Plain ViTs |
|---|---|---|---|---|---|
| first layer | $\text{z}^+$ | WSquare | WSquare | WSquare | $\text{z}^+$ |
| layers inside blocks | $\text{z}^+$ | Gamma | Gamma | Gamma | Gamma |
| activations, layer scale, gates | pass | pass | pass | pass | pass |
| LayerNorm | — | spread constant | spread constant | spread constant | spread constant |
| attention | — | — | weights constant | weights constant | weights constant |
| residual split | equal | proportional | proportional | magnitude | magnitude |

</div>

</details>

## Conclusion
{%- comment -%}TODO: write once the sections are agreed. Open questions grounded in the figures: why ViT maps
barely depend on their start; which residual split to trust; where the grids come from (next post).{%- endcomment %}

LRP does not fail loudly. Where a modern encoder has a part without a rule, the explanation falls back to the gradient and still returns a map. Sometimes it is noise, and sometimes it is a convincing picture built on totals that are orders of magnitude too large or a quarter too small. The pictures alone could not tell these cases apart, but the totals could, because relevance that is only divided should never multiply on its way to the pixels.

All rules from this post live in a small package that comes with the [companion notebook](https://github.com/karray/deep-unboxing) {%- comment -%}TODO: link to 02_... once it exists{%- endcomment -%}. Explaining an encoder takes a few lines.

```python
from xai_utils.models import load_model
from xai_utils.lrp import explain

model = load_model("dino_vit")              # one of the thirteen encoders
x = model.transform(image)[None]            # a 224x224 PIL image -> (1, 3, 224, 224)
relevance = explain(model, x)               # (3, 224, 224), one map per colour channel
heatmap = relevance.clamp(min=0).mean(0)    # positive part, averaged over the colours
```

It runs on the CPU in a few seconds per image and gives identical numbers on every machine. To add a new model, write a canonizer for each of its blocks that makes the residual additions explicit and treats the attention weights and the LayerNorm spread as constants, then run the checks below to see whether anything was missed.

<details markdown="1">
<summary>What was checked and what was not</summary>

A broken rule can still produce a plausible map, so the pictures alone are not enough. These simple checks are worth running for any new model.

- The model's output must not change when the rules are installed, since every rewrite only changes the backward pass.
- Every layer must receive a rule. Counting the layers without one catches cases like the GELU above.
- The relevance reaching the pixels should be plausible and never exceed the starting relevance by orders of magnitude.
- After the explanation, the model must be restored exactly, so that later predictions are not affected.

{%- comment -%}TODO: these checks come from the research code; add them to notebook 02 (and xai_utils) so the claim holds for the published package.{%- endcomment %}

The checks catch broken rules, but several questions remain open.

- The maps depend on the rules. The residual split alone changes how much relevance arrives and where it goes.
- In plain ViTs, the maps barely depend on where the relevance starts, and for the supervised ViT this held under every residual rule tried here. If the starting point hardly matters, it is not obvious what exactly such a map explains.
- Faithfulness was not measured. Readers who need it will find established measures in the work of Achtibat et al. and Berend et al.
- MAE keeps a strong grid in its maps under every first-layer rule tried here, and no other change removed it either. Part of it seems to come from MAE's own learned weights, which favor the borders of each patch, but where exactly it comes from is a question for a separate post.

</details>

With the rules described here, the maps of ConvNeXt, Swin and ViTs make sense again, but a map that makes sense is not yet a faithful one. Two questions from the figures remain. Why do the maps of plain ViTs barely depend on where the relevance starts? And which of the residual splits describes what the model actually does?

## References

{%- capture references -%}
Simonyan, K., Vedaldi, A., & Zisserman, A. | 2014 | Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps;
Springenberg, J. T., Dosovitskiy, A., Brox, T., & Riedmiller, M. | 2015 | Striving for Simplicity: The All Convolutional Net;
Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K.-R., & Samek, W. | 2015 | On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation;
Montavon, G., Binder, A., Lapuschkin, S., Samek, W., & Müller, K.-R. | 2019 | Layer-Wise Relevance Propagation: An Overview;
Lapuschkin, S., Wäldchen, S., Binder, A., Montavon, G., Samek, W., & Müller, K.-R. | 2019 | Unmasking Clever Hans Predictors and Assessing What Machines Really Learn;
Anders, C. J., Neumann, D., Samek, W., Müller, K.-R., & Lapuschkin, S. | 2021 | Software for Dataset-wide XAI: From Local Explanations to Global Insights with Zennit, CoRelAy, and ViRelAy;
Ali, A., Schnake, T., Eberle, O., Montavon, G., Müller, K.-R., & Wolf, L. | 2022 | XAI for Transformers: Better Explanations through Conservative Propagation;
Achtibat, R., et al. | 2024 | AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for Transformers;
Bakish, Y., Zimerman, I., Chefer, H., & Wolf, L. | 2025 | Revisiting LRP: Positional Attribution as the Missing Ingredient for Transformer Explainability;
Berend, J., Achtibat, R., Schäffer, D., Binder, A., Samek, W., Lapuschkin, S., & Dreyer, M. | 2026 | ResLRP: The Role of Residual Cancellation in Attribution Instability in Vision Transformers;
{%- endcapture -%}

{% include reference.html ref=references %}

<script src="{{ '/assets/js/posts/label-free-lrp.js' | relative_url }}" defer></script>
