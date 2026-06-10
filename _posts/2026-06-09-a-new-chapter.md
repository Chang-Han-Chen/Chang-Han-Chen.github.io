---
title: "A New Chapter"
layout: post
date: 2026-06-09
excerpt: "How can one resist the temptation to understand intelligence?"
permalink: /blog/a-new-chapter/
---

*How can one resist the temptation to understand intelligence?*
{: .post-subtitle}

Since this is my very first blog post, I want to briefly explain my interests. We are building human-level — perhaps soon beyond-human-level — intelligence in the lab. I find this mind-blowing. Anyone with the slightest curiosity about how the universe works cannot possibly resist the temptation to understand it. I certainly could not.

## My first astonishment

I should confess that I was dismissive of language models for a long time. I had tried the earlier chatbots on real physics, and they were glib, agreeable, and useless — fancy autocomplete with good manners. I filed the whole thing away and went back to my black holes.

Then OpenAI's o3 came out, and I tried again. The harder the task I gave it, the longer it thought. That was very surprising to me!

My first test was an integral from a small calculation in JT gravity. If you have ever computed an out-of-time-order correlator in JT or SYK, you know the type: an integral over intermediate energies, with a pile of Gamma functions that is supposed to collapse into a hypergeometric function. Doing it by hand costs an afternoon, mostly spent hunting through tables for the identity that makes everything click. o3 thought for thirty seconds and nailed it.

So I escalated to a question from the frontier of my actual research: how can a limit of a von Neumann algebra become a Poisson algebra? It thought for three minutes — and gave me circular hand-waving. It restated my question in fancier language, decorated the restatement with the right vocabulary, and presented it back to me as an answer.

And yet, notice what happened. It did not solve my research question, but it somehow *knew* the second question was much harder than the first, and allocated its effort accordingly. It seemed to understand and appreciate the varying difficulty of the questions. This does not feel like pattern matching to me?!

I was desperate to understand how this works, so I went on a reading spree: chain-of-thought, R1, test-time compute, and so on. The papers were mostly about post-training, but the more I read, the more the post-training miracles looked like *elicitation* rather than creation. Consider the budget: post-training typically spends well under a percent of the total compute and data, far too little to build knowledge from scratch — yet apparently enough to transform behavior. The resolution of this puzzle, I think, is that the behaviors are not being built; they are being *selected*. Run pure RL on a base model and it starts reflecting, backtracking, double-checking — but sample the base model enough times and you find those same patterns already present, just rare. Problems the RL model solves on the first try, the base model often solves within many tries. Post-training sharpens the distribution around capabilities that pretraining already paid for — Nathan Lambert calls this the [elicitation theory of post-training](https://www.interconnects.ai/p/elicitation-theory-of-post-training). So I arrived at the intuition that **pretraining is the foundation of capabilities, from which post-training elicits and forges targeted behaviors** — and if I wanted to understand where the intelligence comes from, I had to start at the foundation.

There is also a practical consideration pointing the same way. The compute barrier for post-training research is high: you need at least enough compute to run inference on, and finetune, a strong-enough base model. Pretraining research is more accessible than it looks. A good architecture or optimizer intervention often shows a clear signal in a scaling-law experiment at the ~100M-parameter scale — well within the reach of a researcher with a few rented GPUs. I will report some of these in later posts.

## Back to school

A few months ago, I started thinking more seriously about deep learning research and decided to take some courses at Berkeley: *CS288 Advanced NLP* with Sewon Min and *Seminar in Deep Learning Theory* with Jason Lee. They were more fun and useful than I expected, and mostly because of the people. Sewon advised my diffusion language model project and helped me get compute support from VESSL AI. With Jason, I chatted about Muon and &mu;P **(TODO: one or two concrete things we discussed)**. There is no substitute for talking to experts about frontier problems.

Through the courses, three topics kept pulling at my attention: diffusion, &mu;P, and Muon. You might notice that these topics all have strong theoretical appeal. Well, of course — I am trained in theoretical physics ;)

**Diffusion** generates by iterative denoising rather than strictly left-to-right prediction. What hooks me is that it offers a different way to leverage compute at test time: instead of serializing all reasoning into a chain of tokens, the model refines a whole block in parallel — something closer to non-verbal reasoning. **&mu;P** reparameterizes a network so that optimal hyperparameters become stable across model scales. It replaces "tune at full scale and pray" with a principled answer to where hyperparameters come from. The spectral-norm version of the analysis is remarkably clean; when I presented it in Jason's seminar, he told me it was the best presentation of the semester. And **Muon** is a matrix-native optimizer: it treats a weight matrix as a matrix, with the geometry that entails, rather than a bag of independent scalars to be updated elementwise.

The diffusion thread has already turned into a real project. I first tested my curriculum idea at 1M-parameter scale on TinyShakespeare, with no funding at all, and once Sewon helped me secure compute from VESSL AI, I scaled it up to 70M MoE models. The same optimal 30% autoregressive fraction showed up at both scales — I still remember how surprising that was to see.

## What my PhD taught me

My PhD taught me one thing: don't set lofty goals — lower my standards. Fantasizing about progress is counterproductive. Just be honest with myself about whether I understand something, explain it until it sounds trivial, and then iterate on that as fast as possible.

[My clock paper](https://arxiv.org/abs/2406.02116) is my best example of this. A famous paper had shown that adding an observer to an idealized universe yields a sensible entropy and a nice algebra, and the community was excited. Many claimed that the same procedure should work in general, despite nobody being able to pin down the details. What Geoff and I did was, in some sense, very pedantic: we followed our noses and did the simplest calculation in the simplest setting we could fully understand. It turned out the procedures we needed were mostly new. Two years later, I still get invitations to workshops and seminars for this paper, and people keep telling me they appreciate the nuances in the calculation — nuances that at the time felt like the most obvious things in the world, things I assumed I was just too slow to be confused by. Turns out I was not, and I am glad I stuck with the confusion until I figured it out.

With that said, if I am forced to state an overarching goal, it would be this:

**How do we make pretraining more predictable, more principled, and more efficient?**
{: .notice--info}

## Wait, why not mech interp?

Before closing, let me address a question I keep getting: "if you want to understand how the model works, why don't you do mechanistic interpretability?" Yes — that is exactly where I started. I was very impressed by Anthropic's papers on [superposition](https://transformer-circuits.pub/2022/toy_model/index.html) and [induction heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html). However, I did not get much out of the later work on SAEs and neuron-level interpretability.

Why so fixated on neurons? To a physicist, this is like studying spin liquids or high-temperature superconductors at the level of individual molecules. They are just not the right units. Sometimes we are fortunate and the system hands us emergent quasi-particles, such as [anyons](https://en.wikipedia.org/wiki/Anyon) — but more often it does not!

Physicists have a useful distinction here: *kinematics* versus *dynamics*. Kinematics is the inventory of a system — what the parts are and what states they can be in. Dynamics is how the system responds when you push it. Think anatomy versus physiology: an anatomical chart tells you where every organ sits, but to understand a heart, you make it beat faster and watch what happens. When a system has no clean quasi-particles, physicists fall back on dynamics: probe it with perturbations, measure the response order by order, and include non-perturbative effects when they matter. To me, cataloguing what each neuron or feature represents is kinematics — anatomy. What I want is the physiology of LLMs: change the scale, the data, the architecture, the optimizer, and watch how the loss and the behavior respond. Scaling laws are exactly such response functions. That, I would later learn, is called pretraining science.

I still spent some time on it, and trained linear probes as lie detectors myself. I learned useful things — how to set up a pipeline for model-organism research and data collection — but I did not learn much about the model's internals.

Then I switched to pretraining research. To my surprise, I feel I have learned more about *why* LLMs work from, say, &mu;P and attention sinks than from all the mech interp papers after Superposition and Induction Heads combined. Of course, this is just my personal experience and research taste — and almost surely, my underwhelming experience with mech interp was a skill issue on my side. There are many fantastic mech interp researchers making real progress on how LLMs work.

## What's next?

I am pivoting my research to AI, and this blog is where I will think out loud. The next post is already up: [a block-size curriculum for diffusion language model pretraining](/blog/how-long-ar-before-diffusion/), my term project for Sewon's class. After that, I plan to write about &mu;P and scaling laws of architecture interventions.

I have to say, it feels fresh to be this ignorant again. In quantum gravity, I know the literature, the people, and the moves. In deep learning, I am a beginner: every week, something I believed turns out to be wrong. Exposing my ignorance and curiosity to the world like this is a bit scary — and so, so exciting.
