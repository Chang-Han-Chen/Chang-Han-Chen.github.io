---
title: "A New Chapter"
layout: post
date: 2026-06-09
excerpt: "Why a quantum gravity PhD cannot resist the temptation to understand intelligence — and how I landed on pretraining science."
permalink: /blog/a-new-chapter/
---

*The first post of a series on my journey into deep learning: why I am pivoting from quantum gravity to pretraining science.*
{: .post-subtitle}

<details class="post-toc" markdown="1">
<summary>Table of Contents</summary>

* TOC
{:toc}
</details>

Since this is my very first blog post, I want to briefly explain my interests. We are building human-level — perhaps soon beyond-human-level — intelligence in the lab. I find this mind-blowing. Anyone with the slightest curiosity about how the universe works cannot possibly resist the temptation to understand it. I certainly could not.

## The o3 moment

For me, the moment was when OpenAI's o3 came out. The harder the task I gave it, the longer it thought. That was very surprising to me!

Two examples stuck with me. I asked it to evaluate an integral involving hypergeometric functions. It thought for thirty seconds and did it correctly. Then I asked it how a limit of a von Neumann algebra can become a Poisson algebra. It thought for three minutes and gave me nonsense. But notice what happened: it did not solve my research question, yet it somehow *knew* the second question was much harder than the first, and allocated its effort accordingly. It seemed to understand and appreciate the varying difficulty of the questions. This does not feel like pattern matching to me?!

I was desperate to understand how it works. I read the papers on [Chain of Thought](https://arxiv.org/abs/2201.11903), [DeepSeek-R1](https://arxiv.org/abs/2501.12948), [test-time scaling](https://arxiv.org/abs/2408.03314), and so on. These papers are mostly about post-training, but reading them, I eventually formed the intuition that **pretraining is still the foundation of capabilities, from which post-training elicits and forges targeted behaviors**.

There is also a practical consideration. The compute barrier for post-training research is high: you need at least enough compute to run inference on, and finetune, a strong-enough base model. Pretraining research is more accessible than it looks. A good architecture or optimizer intervention often shows a clear signal in a scaling-law experiment at the ~100M-parameter scale. I will report some of these in later posts.

## Wait, why not mech interp?

You might be wondering: "if you want to understand how the model works, why don't you do mechanistic interpretability?" Yes — that is exactly where I started. I was very impressed by Anthropic's papers on [superposition](https://transformer-circuits.pub/2022/toy_model/index.html) and [induction heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html). However, I did not get much out of the later work on SAEs and neuron-level interpretability.

Why so fixated on neurons? To a physicist, this is like studying spin liquids or high-temperature superconductors at the level of individual molecules. They are just not the right units. Sometimes we are fortunate and the system hands us emergent quasi-particles, such as [anyons](https://en.wikipedia.org/wiki/Anyon) — but more often it does not! Instead, physicists understand such systems by probing them with perturbations, studying the response order by order, and including non-perturbative phenomena when they matter. The interesting questions are about *dynamics*, not just *kinematics*. To me, neuron-level interpretability feels like kinematics.

I still spent some time on it, and trained linear probes as lie detectors myself. I learned useful things — how to set up a pipeline for model-organism research and data collection — but I did not learn much about the model's internals.

Then I switched to pretraining research. To my surprise, I feel I have learned more about *why* LLMs work from, say, [&mu;P](https://arxiv.org/abs/2203.03466) and [attention sinks](https://arxiv.org/abs/2309.17453) than from all the mech interp papers after Superposition and Induction Heads combined. Of course, this is just my personal experience and research taste — and almost surely, my underwhelming experience with mech interp was a skill issue on my side. There are many fantastic mech interp researchers making real progress on how LLMs work.

## Back to school

A few months ago, I started thinking more seriously about deep learning research and decided to take some courses at Berkeley. They were more fun and useful than I expected! I took *CS288 Advanced NLP* with Sewon Min and *Seminar in Deep Learning Theory* with Jason Lee, and three topics kept pulling at my attention: diffusion, &mu;P, and Muon.

You might notice that these topics all have strong theoretical appeal. Well, of course — I am trained in theoretical physics ;)

**Diffusion** generates by iterative denoising rather than strictly left-to-right prediction. What hooks me is that it offers a different way to leverage compute at test time: instead of serializing all reasoning into a chain of tokens, the model refines a whole block in parallel — something closer to non-verbal reasoning. **&mu;P** reparameterizes a network so that optimal hyperparameters become stable across model scales. It replaces "tune at full scale and pray" with a principled answer to where hyperparameters come from. **Muon** is a matrix-native optimizer: it treats a weight matrix as a matrix, with the geometry that entails, rather than a bag of independent scalars to be updated elementwise.

Three different doors, but I keep feeling they open into the same room.

## What my PhD taught me

My PhD journey taught me one thing: don't set lofty goals — lower my standards. Fantasizing about progress is counterproductive. Just be honest with myself about whether I understand something, explain it until it sounds trivial, and then iterate on that as fast as possible.

With that said, if I am forced to state an overarching goal, it would be this:

**How do we make pretraining more predictable, more principled, and more efficient?**
{: .notice--info}

## The road ahead

I am pivoting my research to AI, and this blog is where I will think out loud. The next post is already up: [a block-size curriculum for diffusion language model pretraining](/blog/how-long-ar-before-diffusion/), my term project for Sewon's class. After that, I plan to write about &mu;P and scaling methodology, and about the architecture interventions (QK-norm, attention gates, value residuals, MoE, and friends) behind the 70M-scale models in that project.

If any of this resonates with you — or you think my physics analogies are completely wrong — I would love to hear from you: [changhanc@berkeley.edu](mailto:changhanc@berkeley.edu).
