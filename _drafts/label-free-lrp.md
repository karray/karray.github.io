---
layout: post
title: What LRP Is and Why It Fails Quietly on Modern Vision Models
styles:
  - /assets/css/posts/label-free-lrp.css
tags: XAI, LRP, CNN, ViT, Self-Supervised Learning
description: When LRP encounters a layer without a rule, it silently falls back to the gradient, turning heatmaps into noise or producing convincing pictures backed by impossible numbers.
image:
  path: /assets/img/posts/label-free-lrp/preview.jpg
  width: 1200
  height: 630
  alt: Relevance streams from a 7 by 7 feature grid branching through intermediate network stages to form a pixel heatmap of a dog and a cat.
---


Image classifiers do not always look where we expect. One well-known model recognised horses by a photographer's copyright tag in the corner of the frame, and only a map of the pixels it relied on revealed the shortcut. Layer-wise relevance propagation (LRP) is a classic way to draw such maps. On modern vision models, however, it can fail without warning, producing either noise or maps that look plausible while carrying the wrong amounts.

<!--more-->

This post shows why and fills in the missing rules, which come with a [companion notebook](https://github.com/karray/deep-unboxing/blob/main/02_label_free_lrp.ipynb).


## What is relevance?

A deep network passes an image through a stack of layers until the last one makes the prediction. We can view each layer as a separate network in its own right, which takes an input, computes an output and passes it to the next layer. Now think of the output of the whole network as water that we pour back through it. At every layer, the water held by each neuron is divided among the neurons that fed into it, and a neuron that contributed more receives more. To know how much each one contributed, we store the intermediate outputs during the forward pass and reuse them on the way back. Whatever reaches a pixel tells us how much that pixel contributed. This water is called relevance, and the procedure is layer-wise relevance propagation [<a href="#bach2015on" data-ref="bach2015on">Bach et al.</a>]. Spread over all the pixels, the relevance forms a heatmap, and a heatmap like this is what caught the horse classifier reading a copyright tag [<a href="#lapuschkin2019unmasking" data-ref="lapuschkin2019unmasking">Lapuschkin et al.</a>].

The figure below runs LRP on a small network that reads handwritten digits. Its left half computes the prediction, and its right half passes the score back to the pixels. You can draw your own digit on the input.

{% include posts/label-free-lrp-water.html %}

Notice how the two halves mirror each other. The backward pass walks the same path as the forward pass, only in the opposite direction, and at every step it relies on what the forward pass stored there to split the relevance. The relevance usually starts with the score of the class being explained, but here it starts with 1.00. Since relevance is only redistributed and never created, the number under each column stays at 1.00 from layer to layer. This property is called conservation. Starting at 1.00 only rescales the numbers and leaves the maps unchanged, yet it makes conservation easy to check, since any relevance that leaked out or appeared from nowhere would pull a column away from 1.00. Hover over a neuron on the right, or tap it, to see how much relevance it holds and where it flows on its way to the pixels.

How we distribute this relevance is decided by a rule, and each kind of layer needs one that fits what it computes. The figure uses the $z^+$ rule, which follows only the contributions that raised the score. It cannot say what pulled a score below zero, so for a class with a negative score the figure starts with 0 and shows an empty map. Choosing these rules well is most of the work, and *Layer-Wise Relevance Propagation: An Overview* describes the common ones and when to use them [<a href="#montavon2019layer-wise" data-ref="montavon2019layer-wise">Montavon et al.</a>].


## Relevance is not a gradient

Most explanation methods for neural networks are built on the gradient, but the gradient answers a different question than relevance. A single neuron is enough to see the difference. Take one with inputs $x_1 = 2$ and $x_2 = 1$, and weights $w_1 = 1.5$ and $w_2 = -1$. It multiplies each input by its weight and adds the results, so it outputs $3 - 1 = 2$. The gradient asks how sensitive the output is to a small nudge in each input, and for this neuron the answer is simply the weights, 1.5 and −1.

The figure below shows this neuron. Like the digit network, it is mirrored around the output. Drag the inputs and the weights on the forward side, and the relevance flows back to the inputs on the other side, red for positive and blue for negative.

<figure class="fig lrp-fig lrp-neuron">
  <div data-lrp-toy="neuron"><p class="lrp-small">This figure needs JavaScript.</p></div>
</figure>

Relevance asks instead how much of the output came from each input. An input contributes its value times its weight, so $x_1$ contributed 3 and $x_2$ contributed −1, while an input of zero would contribute nothing, however large its weight. LRP divides the relevance that arrives in these proportions, and this basic rule is called LRP-0. Starting from the output of 2, it passes 3 back to $x_1$ and −1 to $x_2$, so the input that worked against the output receives negative relevance, and the total stays 2. The $z^+$ rule of the first figure follows only the positive contributions, with $z$ for a contribution and $+$ for its positive part. It gives all of the 2 to $x_1$ and nothing to $x_2$, so relevance never turns negative. Since every layer is made of such neurons, repeating this step layer by layer carries the relevance all the way to the pixels.

In code, LRP runs on the same backward pass that computes gradients, and each rule overrides how its layer passes values back. A layer without a rule raises no error and simply multiplies the incoming relevance by its gradient, here $1.5 \times 2 = 3$ and $-1 \times 2 = -2$, which adds up to 1 instead of the 2 that arrived. We will call this the gradient fallback, and it is behind every failure in this post. The switch in the figure above shows it for any inputs and weights.

The next figure compares the two on the digit network. With the gradient, the total changes from layer to layer and some values turn negative, shown in blue. The pixel map also reaches into the empty background, which contributed nothing to the score. With LRP, every column holds the same total and the background stays empty.

{% include posts/label-free-lrp-water.html gradient=true mode="gradient" %}

The second switch decides whether to keep negative values, giving four methods in total. The plain gradient is ordinary backpropagation, the same backward pass that trains the network, and its pixel map is known as a saliency map [<a href="#simonyan2014deep" data-ref="simonyan2014deep">Simonyan et al.</a>]. LRP-0 and $z^+$ are the two rules from our single-neuron example. Guided backpropagation is the gradient's counterpart to $z^+$, passing back only the positive part of the gradient, yet some blue remains and the totals still change. You can hover over the classes to compare them.

<details markdown="1">
<summary>What is guided backpropagation?</summary>

Guided backpropagation was introduced to produce sharper pictures of what a convolutional network responds to [<a href="#springenberg2015striving" data-ref="springenberg2015striving">Springenberg et al.</a>]. It runs ordinary backpropagation, but at every neuron it keeps only the positive part of the gradient, and only if that neuron was active in the forward pass. Because this filtering happens at the neurons, multiplying by negative weights on the way to the previous layer can create new negative values. Inside the network, the next set of neurons drops those negative values again, but the input pixels have no neurons to filter them out, leaving some blue in the final heatmap. Since it still multiplies by the weights instead of dividing a fixed share, the totals change from layer to layer.
</details>

The two LRP rules treat the classes differently. Since LRP-0 keeps negative contributions, we can explain a class with a negative score by setting its starting relevance to −1. In the figure above, this produces a distinct map for each class. The $z^+$ rule follows only positive contributions, so it starts at 0 when the score is negative. Even for another class with a positive score, using $z^+$ in every layer produces almost the same map as for the prediction, because without negative contributions the backward pass depends mostly on the input and hardly on the class. For that reason, the figure shows the $z^+$ map only for the predicted class and sets the rest to 0.

## The first attempt

The [previous post]({% post_url 2026-10-01-unsupervised-feature-attribution %}) applied LRP to self-supervised vision encoders, which learn from images without any labels. Instead of a score for each class, such an encoder turns an image into an embedding, a vector of numbers that describes the whole image. Inside, it computes a feature vector for each cell of a coarse grid over the image, and the embedding it outputs is simply their average. The relevance can therefore start from these features directly. Only their positive part is used, scaled to add up to 1 as in the digit figures.

<details markdown="1">
<summary>What are features, and how does the relevance start from them?</summary>

What a feature vector stands for depends on the architecture. For a $224 \times 224$ image, a vision transformer (ViT) cuts the input into patches of $16 \times 16$ pixels and turns each patch into a feature vector, called a token. Its grid therefore has $14 \times 14$ cells, and attention lets every token draw on the whole image. In a convolutional network, each feature vector on the grid summarises a region of the image called its receptive field, which grows with every layer. Hierarchical models like ConvNeXt, Swin, and MaxViT shrink the image in stages to a grid of $7 \times 7$ cells. Each feature vector holds several hundred numbers, called channels, for example 768 in ViT-B and 1024 in ConvNeXt-B.

Each positive number in the grid, one for each cell and channel, starts with its value divided by the sum of all positive numbers across the grid, while negative numbers start with 0. The starting relevance then adds up to 1, and whatever arrives at the pixels is the share that made it all the way through. Since the features add up to different totals in different models and images, this scaling also makes the numbers comparable across all of them. We get the exact same result if we start at the pooled embedding, give each channel its share of this positive sum, and pass the relevance back through the averaging layer with the $z^+$ rule.

Only the positive part is used because the features of most modern encoders are signed, so starting from the raw values would let positive and negative numbers largely cancel out. In ConvNeXt, Swin, and plain ViTs, the raw sum of all feature values retains only 1 to 5% of their total magnitude, and for several of these models the sum is negative on almost every image. ResNets do not have this problem, because their features pass through a ReLU and are never negative. Taking the positive part avoids this cancellation and gives zero or negative features no relevance at all.

</details>

To see how LRP behaves on modern architectures, we will test encoders across the current design space: ConvNeXt (a modernised convolutional network), the plain Vision Transformer (ViT), masked autoencoders (MAE), and two hybrids that blend convolutions with attention, Swin and MaxViT. All of them do most of their work in convolutions and fully connected layers. Both simply multiply each input by a weight and add the results, so we will call them linear layers. Two PyTorch libraries already provide ready-made LRP rules for them and for several surrounding operations. [zennit](https://github.com/chr5tphr/zennit) covers classic convolutional networks such as VGG and ResNet [<a href="#anders2021software" data-ref="anders2021software">Anders et al.</a>]. Instead of editing the model's code, it attaches rules to the layers while an explanation runs, and restores it afterwards. [LXT](https://github.com/rachtibat/LRP-eXplains-Transformers), the implementation of AttnLRP, covers language models and the original vision transformer [<a href="#achtibat2024attnlrp" data-ref="achtibat2024attnlrp">Achtibat et al.</a>].

<details markdown="1">
<summary>Why not simply use zennit or LXT?</summary>

Both libraries have rules for linear layers, and these carry over to the encoders here. However, `zennit` has no rules for LayerNorm or attention. LXT does have rules for LayerNorm, GELU, and attention, but it relies on custom model implementations written for a few specific architectures. The encoders we use here come from the [timm](https://github.com/huggingface/pytorch-image-models) library, which defines these blocks with its own code. We therefore implement the missing rules in `zennit`'s way, attaching them directly to the existing layers.

</details>

So we start where the libraries stop, with the usual rules for the linear layers and no rule for anything else. The first column of the figure below shows the result for three encoders, and in all of them the map is noise.

{% include figure-gallery.html id="lrp-rules" data=site.data.galleries.lrp_rules class="wide" %}

The maps show the positive part of the relevance, while the number under each map is how much of the starting relevance of 1 arrives at the pixels. In the first column, that total ranges from almost nothing in ConvNeXt to more than a thousand in the DINOv1 ViT, and neither is possible when relevance is only redistributed. Every part without a rule fell back to the gradient. The two middle columns each add a single rule, and the last column has all of them. Even there, the total falls short of 1, not because a rule is missing, but because part of the relevance stays inside the network, as we will see shortly.

## The missing rules

Beyond their linear layers, these models rely on five operations that off-the-shelf LRP libraries leave uncovered here, namely residual connections, LayerNorm, GELU, layer scale, and self-attention. To see what each missing rule does, the toy ConvNeXt example below has the same parts as the real encoders, and its switches choose between the rule and the gradient fallback for each one. You can also select any block in the diagram to inspect how the relevance flows through it.

{% include posts/label-free-lrp-cnx.html %}

### Residual connections

In a plain network, each layer passes on only its own output. Modern networks wire skip connections around each block of layers. A residual connection is a skip that carries the input of a block forward and adds it to the block's output, which we call the update. Each block thus computes only a change to its input. The running sum that passes from block to block carries the input along with every change added to it and is called the residual stream.

The addition at the end of each block works like a neuron with two weights of 1, so LRP gives the skip and the update the share each contributed. The gradient of an addition, however, is 1 for both inputs, so the fallback hands the full relevance to each of them, and the total doubles at every block, as you can see in the toy ConvNeXt example above.

A real ConvNeXt has 36 residual connections in a row and a ViT has 24, so the copies pile up from block to block. In the third column of the gallery above, LayerNorm already has its rule, but every residual connection copies the relevance. As a result, 450 thousand times the starting relevance reaches the pixels in the DINOv1 ViT, 720 times in ConvNeXt, and almost three trillion times in Swin. Each map is scaled to its own maximum, so none of this shows visually, and in the DINOv1 ViT the objects even look right. The fix is to replace the gradient at every addition with a rule that divides the relevance between the two branches.

### LayerNorm

Every block adds to the residual stream, so the values can grow large on the way through the network. Each block therefore brings its values into a common range with LayerNorm, which subtracts the mean of each position's values and divides them by their spread.
Adding the same number to every input of a LayerNorm raises their mean by that number and leaves the output unchanged, so the gradient of LayerNorm always sums to zero over its inputs. The fallback therefore splits whatever relevance passes through into positive and negative parts of the exact same size, as the LayerNorm card under the toy ConvNeXt example shows.

A real encoder has a LayerNorm in every block, so this happens dozens of times on the way to the pixels. Which channels receive the negative parts is decided by the normalisation more than by the objects in the image. Once the relevance reaches the input, the two signs land on neighbouring pixels, and the map turns into noise. The second column of the gallery from the first attempt shows this, even though every residual connection there already divides the relevance properly. The totals expose that failure, ranging wildly from 0.01 in ConvNeXt to millions in Swin.

The rule treats the spread as a constant on the way back, while the forward pass stays exactly the same. LayerNorm then behaves like a linear layer, and relevance passes through it with the default rule. Conservative propagation proposed this treatment for transformers [<a href="#ali2022xai" data-ref="ali2022xai">Ali et al.</a>], and AttnLRP uses an equivalent rule [<a href="#achtibat2024attnlrp" data-ref="achtibat2024attnlrp">Achtibat et al.</a>]. Looking at the third column, this rule alone brings out the objects in the DINOv1 ViT, though with the inflated totals of the copying residual connections, while ConvNeXt and Swin stay noisy. Only the last column, with all rules in place, displays the objects cleanly across all three models with a plausible total.

### GELU and layer scale

Two linear layers in a row would act like a single one, so an activation function sits between them and bends each value on its own, which lets the network compute more than weighted sums. ResNet uses ReLU, which zeroes out negative values, while transformers and ConvNeXt use GELU, a smooth cousin of ReLU that lets small negative values pass. Layer scale multiplies each channel of a block's output by a learned number before it joins the residual stream.

GELU and layer scale change each value on its own and mix nothing, so LRP passes relevance through them unchanged, which AttnLRP calls the identity rule. The gradient fallback instead multiplies the relevance by the slope of GELU or by the learned numbers of layer scale. For the small negative values that GELU lets through, the slope is small or even negative, so the relevance they carry is mostly lost or flips its sign. Layer scale shrinks some channels, grows others, and flips the sign of a few.

In `zennit`, the identity rule is applied to every standard activation function, but in ConvNeXt `timm` implements GELU with a custom class that `zennit` fails to recognise, falling back to the gradient silently. In ConvNeXt with the rule, 0.66 of the starting relevance reaches the pixels, and without it only 0.48, so about a quarter is lost on the way. Yet the two maps look the same.

### Attention

Transformer blocks use attention in place of the convolution. A transformer represents the image as tokens, one feature vector per patch, and attention lets each token take a weighted average of what the other tokens send, with weights computed anew for every image. This is the one part that the toy ConvNeXt example lacks.

Under the gradient fallback, each token receives relevance in proportion to its attention weight alone, however much or little it sent. Another part flows into the computation of the weights, in positive and negative amounts that cancel. Because the weights of a token add up to 1, the gradient still divides the relevance among the tokens, only by weight and not by contribution. A missing attention rule therefore does far less harm than a missing rule for LayerNorm or the residual connections.

We can treat the weights as constants on the way back, as if they had been given and not computed from the tokens. No relevance then flows into their computation, and the weighted average works like a linear layer, where each token receives relevance in proportion to its contribution, its weight times what it sent. This is the approach of conservative propagation [<a href="#ali2022xai" data-ref="ali2022xai">Ali et al.</a>]. AttnLRP goes further and also passes relevance through the computation of the weights, with rules designed for it, which made its explanations of classifiers more faithful [<a href="#achtibat2024attnlrp" data-ref="achtibat2024attnlrp">Achtibat et al.</a>]. In our encoders, freezing the weights on the backward pass changes the maps only slightly. Without the rule, Swin's maps pick up a few bright squares and the pixel total shifts by up to about 10%.

### Smaller pieces

A few smaller parts needed a rule as well. MaxViT scales its channels with gates that it computes from the input itself, a design called squeeze-and-excitation. The rule holds these gates constant on the way back, so relevance passes through them unchanged, as with layer scale. Batch normalisation, used in older networks, can be folded into the convolution before it, which turns two layers into one without changing the forward pass. Such rewrites of a model are called canonization [<a href="#motzkus2022measurably" data-ref="motzkus2022measurably">Motzkus et al.</a>]. Folding is only exact when the convolution feeds nothing else, so the code checks that before merging.

## Where the relevance goes

With a rule for every part, the maps make sense again, and we can follow the relevance on its way through the network. The figure below does this for five encoders. Each row starts on the left with the final-layer features and tracks the relevance step by step toward the input pixels on the right.

{% include posts/label-free-lrp-stack.html %}

The encoders differ in how far the relevance moves on its way. A convolution only combines neighbouring cells, so in ConvNeXt the relevance stays close to where it started, and about three quarters of it lands on the starting cell and its eight neighbours. Attention connects every token with every other one, so in a ViT the relevance can jump across the image within a single block. In the DINOv1 ViT, only 40 to 50% stays this close.

ResNet-50 adds a pattern of its own. In its middle stages, one cell in every $2 \times 2$ square holds 60 to 70% of the relevance, imprinting a fine checkerboard that has nothing to do with the image [<a href="#odena2016deconvolution" data-ref="odena2016deconvolution">Odena et al.</a>]. It comes from the downsampling shortcut when a stage halves the spatial resolution. The $1 \times 1$ shortcut steps by two cells at a time and reads only one cell in every $2 \times 2$ square, so on the way back the relevance can only return to the cells that were read. By the time it reaches the pixels, the pattern has evened out.

MAE behaves differently, even though it is a ViT as well. Like DINOv1, it starts with relevance spread almost evenly over the image. DINOv1 gathers it onto the objects on the way back, but MAE keeps 70 to 80% on the starting cell and its neighbours, so its maps stay nearly flat at every depth. If we select a single cell on the dog, a faint trace of the rest of the animal appears, far weaker than in DINOv1.

At the pixels, MAE adds a second oddity. A fine grid traces the border of every patch, and about 70% of the relevance lands within two pixels of a border, in a strip that covers only 44% of the image. In DINOv1, the same strip receives a share in line with its area. Since each patch still passes on exactly what it received from the layer above, the grid must form in the very last step, where the first layer spreads a patch's relevance over its pixels.

The two ViTs with the most scattered maps have something else in common. ConvNeXt, ResNet-50 and Swin learned from labelled images, while DINOv1 and MAE learned from the images alone, without any labels. A supervised model only needs the features that tell its classes apart, so it may learn to ignore much of the background. A self-supervised model has no labels to tell it what matters and may keep features of everything in the image, which would spread the relevance more widely. In this figure the two self-supervised models are also the only plain ViTs, so their training and their architecture cannot be told apart.

A scattered or flat map is not necessarily noise, though. As in the [previous post]({% post_url 2026-10-01-unsupervised-feature-attribution %}), every map in the figure adds up all the features at a position into a single number, so it shows where the relevance lands but not what it stands for. Relevance on the grass and the trees may come from features that describe the background as well as others describe the dog. Once the features are added up, a map has no way to say which is which.

<details markdown="1">
<summary>Where the rest of the relevance goes</summary>

How much relevance arrives differs a lot between models, from 0.02 in ResNet-50 to 0.85 in the supervised ViT. The rest stays inside the network. Most layers add a constant, the bias, to their output. The bias is not an input, yet part of the output is due to it, so the rules leave part of the relevance with it, and small stabilising terms in the rules keep a little more. In transformers, the class token and the register tokens, extra tokens that the model learns and adds to the patch tokens, take their share as well.

Is this a flaw? AttnLRP argues that it is not. Following the original LRP paper, it treats a bias as an extra input whose value is always one, so that its relevance is "preserved, rather than lost" [<a href="#achtibat2024attnlrp" data-ref="achtibat2024attnlrp">Achtibat et al.</a>]. One could force all relevance onto the pixels, either by spreading the bias's share over the inputs or by leaving the bias out of the rules, but AttnLRP warns that spreading can make the results numerically unstable and that leaving out the bias can flip their signs. Leaving out the bias kept all of the relevance in a ResNet but produced invalid numbers in ConvNeXt with the rules used here. This post therefore leaves part of the relevance inside the network and reports how much reaches the pixels.

Position embeddings are the one place where the error goes the other way and too much reaches the pixels. Most ViTs add them to the patch tokens with a plain addition that has no rule here, so its gradient hands the full relevance to the patches, including the share that belongs to the positions. Bakish et al. show that position embeddings carry relevance of their own and that ignoring them violates conservation [<a href="#bakish2025revisiting" data-ref="bakish2025revisiting">Bakish et al.</a>].
</details>

## Choices that remain

Once the rules are in place, three choices are left. They decide how the relevance is split at each residual connection, which rule the linear layers inside the blocks use, and how the first layer spreads relevance over the pixels. None of them breaks the maps, but each one changes them.

### Who gets the credit at a residual connection

A residual connection with a rule must still decide how much of the relevance goes to the skip and how much to the update. There are several reasonable answers, and they behave very differently.

- An **equal** split gives each part half, regardless of the numbers.
- A **proportional** split gives each part the share it contributed. This is the standard LRP treatment of an addition, and it is what LXT and AttnLRP do implicitly. A part that worked against the sum receives negative relevance.
- A **magnitude** split divides by size and ignores the sign. The code uses it for MaxViT and the ViTs, because the proportional split made some of their maps unstable in early experiments.
- A **$\gamma$ split** works like the proportional one, but gives extra weight to the part whose sign agrees with the sum [<a href="#berend2026reslrp" data-ref="berend2026reslrp">Berend et al.</a>].

The proportional, magnitude, and $\gamma$ splits agree as long as the skip and the update have the same sign. They part ways when the update works against the skip. The proportional split then gives the update a negative share, which records that it pulled the sum down. But when the two almost cancel, this split divides by a sum close to zero and hands out huge amounts of both signs. The magnitude split ignores the signs and so avoids this blow-up, but it credits the update as if it had helped. The $\gamma$ split takes the middle way. It keeps the update's negative share but gives extra weight to the part that agrees with the sum, so the two can no longer cancel out and the amounts stay small. Berend et al. see this kind of cancellation inside transformer blocks as a main reason why LRP maps of ViTs turn noisy.

{% include figure-gallery.html id="lrp-residual" data=site.data.galleries.lrp_residual class="wide" %}

In the maps, the equal split stands out first. In a deep network, it sends half of the relevance into every block's update, where part of it stays behind with the biases, so very little reaches the pixels, about 0.04 for the DINOv1 ViT and less than 0.001 for MaxViT. The proportional split delivers the most relevance, but in DINOv1 it also spreads it over the grass and the trees, while the magnitude and $\gamma$ splits keep more of it on the animals. Of these two, the $\gamma$ split passes on more relevance and is a little more detailed.

We can choose a split by looking at maps like these and by checking that small changes to the input do not change the maps much. Which split is more faithful to the model remains untested here, and a handful of images would not be enough to claim one. For anyone who wants to choose between them, Berend et al. compare the $\gamma$ split with the standard rules on several faithfulness measures, which test whether the pixels a map highlights are the ones the model actually depends on.

<details markdown="1">
<summary>A few numbers behind the maps</summary>

The numbers come from eight ImageNet images per model and measure three things for each rule. The first is how much relevance reaches the pixels. The second is how much the map changes when the input moves by a tiny amount. The third is whether the map follows where the relevance starts, checked by starting only from the features inside the object box and then only from those outside.

The equal split delivered the least relevance (0.00 to 0.12 for most models), and its maps barely depended on where they started. The proportional split delivered the most (up to 0.97), followed its starting features most closely, and stayed stable on these images. The magnitude and $\gamma$ splits were the most stable, and the $\gamma$ split followed its starting features almost as closely as the proportional one. The maps of the supervised ViT barely depended on their starting features under any rule.

</details>

### The rule inside the blocks

The linear layers inside the blocks had a rule from the start, but `zennit` offers several, and they trade detail against focus. Apart from ResNet-50, the encoders here use the Gamma rule, which lies between LRP-0 and the $z^+$ rule. Like LRP-0, it keeps negative contributions, but it gives the positive ones extra weight, and the larger its parameter $\gamma$, the closer it comes to $z^+$. This makes the maps less noisy, and AttnLRP recommends it for the linear layers of vision transformers [<a href="#achtibat2024attnlrp" data-ref="achtibat2024attnlrp">Achtibat et al.</a>].

### The last step to the pixels

The first layer of many modern encoders cuts the image into patches and turns each patch into a feature vector, four by four pixels in ConvNeXt and Swin and sixteen by sixteen in a ViT. Its rule decides how a patch's relevance is spread over its pixels, and with that, most of the fine detail in the map. Several rules in `zennit` can handle this step, and it is not obvious which one suits a patch layer.

- **$z^+$** splits a patch's relevance by how much each pixel pushed the layer's output up, as in the digit network.
- **WSquare** splits it by the weights alone, so the image itself plays no part. It is recommended for inputs without natural bounds [<a href="#montavon2019layer-wise" data-ref="montavon2019layer-wise">Montavon et al.</a>].
- **Gamma** also counts the pixels that pushed the output down, but favours those that pushed it up. It is the rule the encoders already use for their linear layers deeper in the network.

Because WSquare ignores the pixels, it fills every patch with the same pattern, scaled by the patch's relevance. In the DINOv2 and DINOv3 ViTs, this one repeated pattern accounts for more than 99% of the detail within the patches. $z^+$ follows the image inside each patch instead, which is why it is the first-layer rule for the ViTs.

{% include figure-gallery.html id="lrp-stems" data=site.data.galleries.lrp_stems_rules %}

Look closely at the full-resolution maps and the patch grid bleeds through. WSquare outlines the patch borders sharply, $z^+$ softens them, and Gamma erases them almost completely. The number under each map measures this as the contrast between neighbouring pixels across a patch border compared with inside a patch, where 1 means no edges. Over the five images, the median for ConvNeXt is 2.11 with WSquare, 1.66 with $z^+$, and 0.97 with Gamma, and Swin and the DINOv1 ViT behave the same way.

MAE is the exception. Its grid survives all three rules, and Gamma leaves it almost as strong as $z^+$ does. Its first layer looks unusual too, putting more than half of its squared weights on the outermost pixels of each patch. These pixels make up less than a quarter of a patch, and in DINOv1 and the supervised ViT they receive a share in line with their number. Even the plain gradient shows the same grid, so it is not an artefact of LRP.

<details markdown="1">
<summary>All rules at a glance</summary>

<div class="lrp-scroll" markdown="1">
The table below lists the rules chosen early on, WSquare for ConvNeXt, Swin, and MaxViT and $z^+$ for the ViTs. By these numbers, Gamma would be the better choice for all of them except MAE, where no rule removes the grid.

| | ResNets | ConvNeXt | Swin | MaxViT | Plain ViTs |
|---|---|---|---|---|---|
| first layer | $z^+$ | WSquare | WSquare | WSquare | $z^+$ |
| layers inside blocks | $z^+$ | Gamma | Gamma | Gamma | Gamma |
| activations, layer scale, gates | pass | pass | pass | pass | pass |
| LayerNorm | — | spread constant | spread constant | spread constant | spread constant |
| attention | — | — | weights constant | weights constant | weights constant |
| residual split | equal | proportional | proportional | magnitude | magnitude |

</div>

</details>

## Conclusion

Maps like these do more than catch a shortcut like the horses at the start. They can also point to quirks of the architecture itself. The checkerboard in ResNet-50 traces back to how its strided layers are wired, and ResNet-D, a later variant, removes the same weakness by averaging before the shortcut [<a href="#he2019bag" data-ref="he2019bag">He et al.</a>]. MAE's maps stay nearly flat and carry a grid along its patch borders that no first-layer rule removes. What causes these patterns, and whether they matter for the model, is a story for another day.

LRP does not fail loudly. Where a modern encoder has a part without a rule, the explanation falls back to the gradient and still returns a map. Sometimes it is noise, and sometimes it is a convincing picture built on totals that are orders of magnitude too large or a quarter too small. Visual inspection alone cannot tell a healthy map from a broken one. The totals can, because relevance that is only divided should never multiply on its way to the pixels. With the rules described here, the maps of ConvNeXt, Swin, and ViTs make sense again.

## References

{%- capture references -%}
Simonyan, K., Vedaldi, A., & Zisserman, A. | 2014 | Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps;
Springenberg, J. T., Dosovitskiy, A., Brox, T., & Riedmiller, M. | 2015 | Striving for Simplicity: The All Convolutional Net;
Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K.-R., & Samek, W. | 2015 | On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation;
Odena, A., Dumoulin, V., & Olah, C. | 2016 | Deconvolution and Checkerboard Artifacts;
Montavon, G., Binder, A., Lapuschkin, S., Samek, W., & Müller, K.-R. | 2019 | Layer-Wise Relevance Propagation: An Overview;
Lapuschkin, S., Wäldchen, S., Binder, A., Montavon, G., Samek, W., & Müller, K.-R. | 2019 | Unmasking Clever Hans Predictors and Assessing What Machines Really Learn;
He, T., Zhang, Z., Zhang, H., Zhang, Z., Xie, J., & Li, M. | 2019 | Bag of Tricks for Image Classification with Convolutional Neural Networks;
Anders, C. J., Neumann, D., Samek, W., Müller, K.-R., & Lapuschkin, S. | 2021 | Software for Dataset-wide XAI: From Local Explanations to Global Insights with Zennit, CoRelAy, and ViRelAy;
Ali, A., Schnake, T., Eberle, O., Montavon, G., Müller, K.-R., & Wolf, L. | 2022 | XAI for Transformers: Better Explanations through Conservative Propagation;
Motzkus, F., Weber, L., & Lapuschkin, S. | 2022 | Measurably Stronger Explanation Reliability via Model Canonization;
Achtibat, R., et al. | 2024 | AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for Transformers;
Bakish, Y., Zimerman, I., Chefer, H., & Wolf, L. | 2025 | Revisiting LRP: Positional Attribution as the Missing Ingredient for Transformer Explainability;
Berend, J., Achtibat, R., Schäffer, D., Binder, A., Samek, W., Lapuschkin, S., & Dreyer, M. | 2026 | ResLRP: The Role of Residual Cancellation in Attribution Instability in Vision Transformers;
{%- endcapture -%}

{% include reference.html ref=references %}

<script src="{{ '/assets/js/posts/label-free-lrp.js' | relative_url }}" defer></script>
