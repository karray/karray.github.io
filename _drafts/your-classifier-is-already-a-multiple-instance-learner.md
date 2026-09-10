---
layout: post
title: "Your classifier sees more than its answer tells you"
description: "A visual guide to global average pooling, multiple-instance learning, and class evidence hidden by an image-level prediction."
math: true
tags: interpretability image-classification multiple-instance-learning GAP
---

A photograph shows a dog beside a bicycle. You ask an image classifier what it sees. It answers: “bicycle.”

Did it miss the dog? Perhaps. But there is another possibility: evidence for the dog survived inside the network and lost when the model combined everything into one answer.

In the previous post, we explored LaFAM: a way to highlight feature activity without choosing a class to explain. That leaves a natural question. Once a region lights up, can we read **which classes the model associates with it**? And can that reading disagree with the final answer?

<!--more-->

<link rel="stylesheet" href="{{ '/assets/css/gap-mil.css' | relative_url }}">
<script defer src="{{ '/assets/js/gap-mil.js' | relative_url }}"></script>

This post follows [*Rethinking Global Average Pooling: Your Classifier Is Secretly a Multi-Instance Learner*](https://arxiv.org/abs/2606.14555). We will unpack the averaging operation at the end of many classifiers and see what it lets us inspect. The interactive examples below use invented scores so you can follow every calculation; the research results are identified separately.

## Before the answer, there is a grid

An image classifier usually does not jump straight from pixels to a word. First, an encoder turns the picture into a collection of numbers called **features**. In the models we are discussing, these features still have spatial positions: a small grid of vectors, each vector being a list of learned measurements.

Think of a group of observers writing notes about a scene. One report may contain evidence about fur and ears, another about wheels, another about the surrounding road. These are an analogy for learned features, not a claim that individual feature coordinates have such tidy meanings.

The reports also overlap. A vector at one position can contain information from a much larger part of the image. A location in the feature grid is not an independently classified crop.

Many classifiers combine the reports using **global average pooling**, or **GAP**. They average each feature across all positions, producing one summary vector. A final **linear classification head** converts this summary into one score per class. These scores are called *logits*. The highest score determines the predicted class.

That sounds like the spatial information has disappeared. But look at what happens if we change the order of the last two operations.

## Read first, average second

Suppose a head turns a feature value into a score by multiplying it by two and adding one. If two locations contain feature values of 1 and 3, we can calculate the answer either way:

<figure class="mil-figure">
  <div class="mil-kicker">01 / Two routes, one answer</div>
  <div class="mil-routes">
    <div><strong>Average, then score</strong><span class="mil-step">Features: 1 and 3</span><span class="mil-arrow">↓ average</span><span class="mil-step">Summary: 2</span><span class="mil-arrow">↓ multiply by 2, add 1</span><span class="mil-answer">Score: 5</span></div>
    <div><strong>Score, then average</strong><span class="mil-step">Features: 1 and 3</span><span class="mil-arrow">↓ multiply each by 2, add 1</span><span class="mil-step">Local scores: 3 and 7</span><span class="mil-arrow">↓ average</span><span class="mil-answer">Score: 5</span></div>
  </div>
  <figcaption>The second route keeps the individual scores available. The final answer is identical. The same calculation works with vectors and many classes.</figcaption>
</figure>

The useful fact is that a linear head, including its bias, **commutes with averaging**: either order gives the same result. Applying the existing head at every grid position therefore reveals a spatial score for every class without training another classifier.

<details class="mil-details" markdown="1">
<summary>The whole identity in one equation</summary>

Let $h_i$ be the feature vector at position $i$, with $N$ positions. For class $c$, the head has weights $w_c$ and bias $b_c$. Then

$$
z_c = w_c^\top\left(\frac{1}{N}\sum_i h_i\right)+b_c
    = \frac{1}{N}\sum_i\underbrace{(w_c^\top h_i+b_c)}_{\text{spatial score }s_{ic}}.
$$

The bias is included at every position and averaged, so it still contributes exactly once. “Linear head” is common shorthand here; with a bias, the precise term is *affine*.

This identity applies to a spatial grid followed by mean pooling and an affine head. A nonlinear operation between pooling and the head needs separate treatment. Nor does the equation automatically describe a transformer that predicts from a special class token.

</details>

There is a related distinction that matters in practice. **Average the scores, not their probabilities.** Softmax, the operation that turns a vector of scores into probabilities, is nonlinear. Applying it at each location and averaging the resulting probabilities generally changes the model's answer.

For these architectures, the class-weighted spatial sums are also the basis of [Class Activation Mapping (CAM)](https://arxiv.org/abs/1512.04150). We can inspect one class map, or compare all class scores at every location to construct a grid of local winners. The multiple-instance perspective gives us a way to think about how these spatial readings combine into the image prediction.

## An image as a bag of evidence

In **multiple-instance learning (MIL)**, each labeled training example is a collection—a *bag*—of instances. The collection has a label, but its members need not have individual labels.

Imagine receiving a box of assorted fruit with the label “contains an apple.” You know something about the box, but nobody has marked which piece is the apple. Learning from that kind of supervision requires connecting a collection-level label to evidence within the collection.

For our classifier, the image is the bag, the spatial feature vectors are the instances, and GAP is the rule for combining their scores. This is **mean-aggregation MIL**. The familiar “one positive instance is enough” rule is another MIL formulation; an average does not implement that rule.

The observers analogy makes the difference clear. “Did anyone report a dog?” and “Which class has the highest average score?” are different questions. A classifier answering the second can lose evidence that would answer the first.

## A small object can lose the average

Consider a toy grid with 36 positions. At object positions, the dog score is 6 and the bicycle score is 0. Elsewhere, the dog score is 0 and the bicycle score is 2. Those other positions might represent another object or context favoring the competing class.

Every object position favors “dog.” Yet if there are only six such positions, the average dog score is 1 and the average bicycle score is about 1.67. The final prediction is “bicycle.”

<figure class="mil-figure" id="mil-area">
  <div class="mil-kicker">02 / When local evidence loses</div>
  <p class="mil-instruction">Change how many positions support the dog. Their individual scores stay fixed.</p>
  <label for="mil-count">Dog-supporting positions: <output id="mil-count-value" for="mil-count">6 of 36</output></label>
  <input id="mil-count" type="range" min="1" max="35" value="6" disabled>
  <div class="mil-demo-columns">
    <div><div id="mil-grid" class="mil-grid" role="img" aria-label="36 positions: six support dog and thirty support bicycle"><span class="mil-bike-cell" aria-hidden="true">B</span><span class="mil-bike-cell" aria-hidden="true">B</span><span class="mil-bike-cell" aria-hidden="true">B</span><span class="mil-bike-cell" aria-hidden="true">B</span><span class="mil-bike-cell" aria-hidden="true">B</span><span class="mil-bike-cell" aria-hidden="true">B</span><span class="mil-bike-cell" aria-hidden="true">B</span><span class="mil-bike-cell" aria-hidden="true">B</span><span aria-hidden="true">D</span><span aria-hidden="true">D</span><span class="mil-bike-cell" aria-hidden="true">B</span><span class="mil-bike-cell" aria-hidden="true">B</span><span class="mil-bike-cell" aria-hidden="true">B</span><span class="mil-bike-cell" aria-hidden="true">B</span><span aria-hidden="true">D</span><span aria-hidden="true">D</span><span class="mil-bike-cell" aria-hidden="true">B</span><span class="mil-bike-cell" aria-hidden="true">B</span><span class="mil-bike-cell" aria-hidden="true">B</span><span class="mil-bike-cell" aria-hidden="true">B</span><span aria-hidden="true">D</span><span aria-hidden="true">D</span><span class="mil-bike-cell" aria-hidden="true">B</span><span class="mil-bike-cell" aria-hidden="true">B</span><span class="mil-bike-cell" aria-hidden="true">B</span><span class="mil-bike-cell" aria-hidden="true">B</span><span class="mil-bike-cell" aria-hidden="true">B</span><span class="mil-bike-cell" aria-hidden="true">B</span><span class="mil-bike-cell" aria-hidden="true">B</span><span class="mil-bike-cell" aria-hidden="true">B</span><span class="mil-bike-cell" aria-hidden="true">B</span><span class="mil-bike-cell" aria-hidden="true">B</span><span class="mil-bike-cell" aria-hidden="true">B</span><span class="mil-bike-cell" aria-hidden="true">B</span><span class="mil-bike-cell" aria-hidden="true">B</span><span class="mil-bike-cell" aria-hidden="true">B</span></div><p class="mil-legend"><span class="mil-dog-text">D: dog score 6</span><br><span class="mil-bike-text">B: bicycle score 2</span><br>Other class score: 0</p></div>
    <div class="mil-score-panel"><div class="mil-score-label">Mean dog score <strong id="mil-dog-value">1.00</strong></div><div class="mil-track"><div id="mil-dog-bar" class="mil-bar mil-dog-bar" style="width:16.667%"></div></div><div class="mil-score-label">Mean bicycle score <strong id="mil-bike-value">1.67</strong></div><div class="mil-track"><div id="mil-bike-bar" class="mil-bar mil-bike-bar" style="width:27.778%"></div></div><p id="mil-area-result" class="mil-result" aria-live="polite">GAP predicts bicycle. All 6 dog positions still predict dog locally.</p></div>
  </div>
  <noscript><p>Static example: six dog positions contribute 6 × 6 / 36 = 1.00; thirty bicycle positions contribute 30 × 2 / 36 = 1.67. Enable JavaScript to change the proportions.</p></noscript>
  <figcaption>A controlled score-grid illustration, not model inference. Increasing the dog region changes its share of the average. In a real image, resizing an object can also change the features themselves.</figcaption>
</figure>

A helpful quantity is the **margin**: dog score minus bicycle score. Here it is +6 at dog positions and −2 elsewhere. If the dog occupies a fraction $a$ of the grid, its image-level margin is

$$
6a - 2(1-a) = 8a - 2.
$$

It wins only when $a>1/4$. At exactly one quarter, the scores tie.

This is about both **strength and extent**. A few strong scores can be outweighed by many weaker scores for a rival. Conversely, more positions can carry the correct class to victory.

There is an important boundary to this example. Adding positions with zero scores for *both* classes would shrink the margin toward zero, but would not reverse its sign. To flip the ranking here, the extra positions must favor the competitor. “Background dilutes the object” is incomplete unless we ask what happens to the competing scores too.

## Why not listen to the loudest position?

If averaging loses a small object, taking the maximum score sounds attractive. One strong position would be enough. But it would also be enough if that position were wrong.

<figure class="mil-figure" id="mil-outlier">
  <div class="mil-kicker">03 / The loudest observer can be mistaken</div>
  <p>Eight positions support dog with score 2. One other position supports bicycle with the score below. All remaining class scores are zero.</p>
  <label for="mil-spike">One bicycle score: <output id="mil-spike-value" for="mil-spike">5</output></label>
  <input id="mil-spike" type="range" min="0" max="20" step="1" value="5" disabled>
  <div class="mil-votes" aria-label="Eight dog positions and one bicycle position"><span>D</span><span>D</span><span>D</span><span>D</span><span>D</span><span>D</span><span>D</span><span>D</span><span class="mil-bike-cell">B</span></div>
  <div class="mil-routes"><div><strong>Mean pooling</strong><p id="mil-mean-result" aria-live="polite">Dog: 1.78 · Bicycle: 0.56 → dog</p></div><div><strong>Max pooling</strong><p id="mil-max-result" aria-live="polite">Dog: 2.00 · Bicycle: 5.00 → bicycle</p></div></div>
  <figcaption>Invented scores, holding the feature evidence fixed. Max pooling is sensitive to a single extreme score; mean pooling can also flip if that score becomes large enough. Neither rule knows whether the unusual position is a rare object or a mistake.</figcaption>
</figure>

Changing the pooling rule changes what evidence is rewarded. It can be useful, but it is a model change to evaluate, not a free correction obtained from the identity above. Looking inside a classifier helps identify the problem before deciding how to change it.

## How can it learn several objects from one label?

Return to the photograph of a dog and a bicycle. A dataset may attach only “dog,” even though “bicycle” is also true. During training, a standard single-label loss rewards the annotated answer and penalizes competing answers. It does not hand the model a separate label for every object.

Across many images, however, objects can appear in different combinations. Dogs occur with bicycles, sofas, people, and sometimes alone. A consistently useful dog-related feature can emerge despite the incomplete labels. This is a possibility of learning across the dataset, not a guarantee for every object or every training distribution. If dogs always appear with the same sofa, the label alone does little to distinguish them.

A simple thought experiment separates visual recognition from the scoring rule. Suppose every picture contains four different shapes, and an annotator randomly chooses one of them as the label. Even a system that recognizes all four cannot predict that random choice better than 25%, if it must return one answer and has no information about the choice. Low single-label accuracy would then coexist with excellent recognition of the visible objects.

The paper tests this idea with images containing three to five shapes and one sampled target. Models trained from scratch reach about 26% image-level accuracy, while their spatial readings identify visible objects much more often. See [the synthetic experiment, Sections 4.4–5](https://arxiv.org/html/2606.14555v1#S4.SS4).

The lesson is to ask what an error actually measures. Failure to choose the one recorded label, failure to recognize an object, and failure to locate it are different failures.

## What shows up in real classifiers?

The paper also checks pretrained ImageNet classifiers against object bounding boxes. For ResNet-50, the image prediction is correct on 80.3% of images; the target wins at least one foreground grid cell on 94.3%. Yet only 33.7% of foreground cells predict the target. These are three different measurements, shown below. [Source: Table 1](https://arxiv.org/html/2606.14555v1#S5.T1).

<figure class="mil-figure">
  <div class="mil-kicker">04 / A correct answer somewhere is not a complete object map</div>
  <p><strong>ResNet-50 · ImageNet validation</strong></p>
  <div class="mil-evidence-row"><span>Correct image prediction</span><strong>80.3%</strong><div class="mil-track"><div class="mil-bar mil-neutral-bar" style="width:80.3%"></div></div></div>
  <div class="mil-evidence-row"><span>Target wins somewhere in foreground</span><strong>94.3%</strong><div class="mil-track"><div class="mil-bar mil-dog-bar" style="width:94.3%"></div></div></div>
  <div class="mil-evidence-row"><span>Foreground cells predicting target</span><strong>33.7%</strong><div class="mil-track"><div class="mil-bar mil-bike-bar" style="width:33.7%"></div></div></div>
  <figcaption>Reported measurements, not toy scores. The first two metrics count images; the third measures foreground-cell accuracy. Bounding boxes define foreground for evaluation. These bars do not represent interchangeable accuracy estimates.</figcaption>
</figure>

Finding the target somewhere gives the model many chances to succeed. It does not tell us how many wrong classes also appear, which region to trust without annotations, or whether the whole object is covered. The higher number is therefore evidence of recoverable spatial information, not the accuracy of an improved classifier.

For the same reason, a local winning class is a diagnostic reading, not a calibrated statement that an object occupies that cell. Coarse grids, shared object parts, and context all complicate that interpretation.

## What to look at when the answer is wrong

The next time a classifier returns a surprising label, its spatial scores offer a more specific set of questions:

- Does the expected class win anywhere, or is its evidence weak throughout the grid?
- Is a competing class concentrated on another visible object, or spread across much of the scene?
- Does the dataset's single label leave out something the model correctly recognized?

LaFAM supplied a class-agnostic view of feature activity. Applying the classifier at each position adds class-specific readings. GAP then explains exactly how those readings become one set of image scores, for the architecture we have considered.

A remaining question is what a spatial vector actually knows about the rest of the image. A score at a background position might depend on an object elsewhere. We will explore that in the next post, on **CNN versus ViT mechanisms**: how information reaches different positions, and what that means when we interpret their maps.

<!-- Editorial: the preceding LaFAM post is currently on branch `lafam`, at
_posts/2025-11-01-unsupervised-feature-attribution.md. Add its final URL once
that post's publication date and slug are fixed. This draft intentionally
uses no link to an unpublished page. -->
