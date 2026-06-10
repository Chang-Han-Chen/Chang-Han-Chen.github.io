---
title: "How Long Should We Train AR Before Switching to Diffusion?"
date: 2026-06-10
categories:
  - ai
tags:
  - diffusion language models
  - pretraining
  - transformers
  - mixture of experts
  - muon
excerpt: "Every at-scale diffusion LM spends 85–99.9% of its budget on autoregressive pretraining. I swept the whole range at 70M scale and found the optimum near 30%."
permalink: /blog/how-long-ar-before-diffusion/
classes: wide
author_profile: false
---

*A small-scale experiment on whether diffusion language models should really be initialized from almost-fully-trained autoregressive checkpoints.*
{: .post-subtitle}

<details class="post-toc" markdown="1">
<summary>Table of Contents</summary>

* TOC
{:toc}
</details>

This is the first post in what I hope becomes a small series about my journey into deep learning. I started seriously studying it only a few months ago. This semester I took *Advanced NLP* with Sewon Min and *Seminar in Deep Learning Theory* with Jason Lee, and three topics kept pulling at my attention: diffusion language models, &mu;P, and Muon. They felt like three doors into the same larger question: **how do we make pretraining more predictable, more principled, and less like folklore?**

This post is about the first door. It covers my term project for Sewon's class, where I studied a simple curriculum for pretraining block-diffusion language models. ([Full report PDF here.](/assets/pdf/block-size-curriculum-dlm-report.pdf))

## The question nobody swept

Diffusion language models (dLMs) are expensive to pretrain, so essentially every recent at-scale dLM is built by *converting* a pretrained autoregressive (AR) model: take a strong AR checkpoint, then run a comparatively brief diffusion-adaptation phase. If we define $$p_{\mathrm{AR}}$$ as the fraction of the total training budget spent on the AR objective, current practice looks like this:

| Model | Recipe | Effective source fraction |
|---|---|---|
| BD3LM | 850K MDLM steps + 150K block-diffusion steps | $$p_{\mathrm{MDLM}} = 0.85$$ |
| Dream-7B | Qwen2.5-7B (~18T tokens) + 580B diffusion tokens | $$p_{\mathrm{AR}} \approx 0.97$$ |
| DiffuLLaMA | LLaMA-2 (~2T) + ~65B adaptation tokens | $$p_{\mathrm{AR}} \approx 0.97$$ |
| SDAR | Qwen3 (~36T) + 50B adaptation tokens | $$p_{\mathrm{AR}} \approx 0.999$$ |

This is a perfectly sensible thing to do *when the AR checkpoint already exists*. But it leaves the middle of the $$p_{\mathrm{AR}}$$ range completely unexplored, and it smuggles in an assumption:

**If we actually believed the final model should be a diffusion model, would we still spend 97–99.9% of pretraining on the AR objective?**
{: .notice--info}

My answer, at the scale I could afford to test: probably not. In my sweep, **spending about 30% of the budget on AR and then switching to small-block diffusion** beat both training diffusion from scratch and the AR-heavy conversions everyone uses.

## Why I care about this

I am very interested in pretraining science, which is a slightly dangerous interest for an independent researcher — the cleanest version of many questions seems to require a datacenter. But I think there is another way to make progress: turn a frontier-scale question into a small, testable hypothesis. The result may not settle the frontier question, but it can tell you whether an idea has the right shape. That was the spirit here. I didn't try to train a giant dLM; I tried to isolate one knob that current recipes leave implicit: $$p_{\mathrm{AR}}$$.

## First, a model I trusted

Before asking curriculum questions, I needed to learn to train a modern-ish transformer at all. I started from a LLaMA-style 70M transformer and layered in interventions from recent work: QK-norm, per-head attention gates, depth-scaled layernorm, value residuals, ReGLU, RMSNorm, and interleaved MoE layers. The optimizer is NorMuon + AdamW: Muon handles matrix parameters, AdamW handles embeddings, the tied LM head, and scalar parameters (Adam betas $$(0.95, 0.99)$$, 100-step warmup, then constant LR).

A side question I wanted to answer: most architecture papers study AR models — do the same tricks help once the objective changes to diffusion? My short answer: **probably yes**. On the AR objective the combined recipe improved best validation loss by about 3.5% over my plain baseline, and the same architecture carried over to the diffusion runs without issues. This part wasn't the headline result, but a curriculum experiment is only meaningful if the underlying model isn't falling apart for unrelated reasons — and it's where I learned to train MoE models, watch router health, and read what optimizer choices do to loss curves.

## The curriculum knob

Block diffusion (BD3) gives a clean way to frame the problem. AR is block size $$b = 1$$: predict left to right. Full masked diffusion (MDLM) is block size $$b = L$$: denoise the whole sequence at once. BD3 sits in between: decode blocks left to right, but do diffusion-style denoising *within* each block. So a two-stage curriculum is parameterized by $$(p_{\mathrm{AR}}, b)$$:

1. Train autoregressively (block size 1) for the first $$p_{\mathrm{AR}}$$ of the optimizer-step budget.
2. Switch to BD3 with block length $$b$$ for the rest.

Existing A2D conversion is exactly this curriculum with $$p_{\mathrm{AR}}$$ pinned near 1, and scratch BD3 is $$p_{\mathrm{AR}} = 0$$. I swept the middle: $$p_{\mathrm{AR}} \in \{0.30, 0.50, 0.80, 0.90, 0.95\}$$ crossed with post-switch block lengths $$b \in \{4, 16, 64, 256\}$$, at 70M MoE scale on ClimbMix (sequence length 512, batch 512, 10,100 optimizer steps ≈ 1.54B unique tokens). I also ran the MDLM-init analogue, where the source phase is full-sequence diffusion instead of AR.

## Result 1: the optimum is at 30% AR

Best validation loss over the full run, AR-init grid (bold = best per column):

| $$p_{\mathrm{AR}}$$ | $$b=4$$ | $$b=64$$ | $$b=256$$ |
|---|---|---|---|
| 0.00 (scratch) | 2.7364 | 2.9467 | **2.9408** |
| 0.30 | **2.7275** | **2.9179** | 2.9629 |
| 0.50 | 2.7717 | — | 3.0124 |
| 0.80 | 2.8511 | — | 3.1512 |
| 0.90 | 2.8786 | — | 3.2469 |
| 0.95 | 2.9924 | — | 3.4385 |

Three things to notice. For every populated row, the best block length is $$b = 4$$. For every column except $$b = 256$$, the best switch point is $$p_{\mathrm{AR}} = 0.30$$ — it beats scratch *and* beats every AR-heavy curriculum. And the AR-heavy corner where current practice lives ($$p_{\mathrm{AR}} \ge 0.8$$) is the *worst* part of the populated range, monotonically degrading as $$p_{\mathrm{AR}}$$ grows.

<figure>
  <img src="/assets/images/ar_vs_scratch.png" alt="AR-init vs scratch BD3 eval-loss curves at block lengths 256, 64, and 4">
  <figcaption>AR-init vs. scratch BD3 at three block lengths. Left: at b=256 scratch beats every AR-init run. Middle: at b=64, p_AR=0.30 beats scratch by 0.029 nats. Right: at b=4, p_AR=0.30 wins by 0.009 nats.</figcaption>
</figure>

The MDLM-init track tells the same story: among MDLM→BD3 runs, $$p_{\mathrm{MDLM}} = 0.30$$ wins at every $$b \le 64$$, though it transfers less cleanly than AR-init does (at $$b = 4$$ it loses to scratch by 1.6%). Interpolating my grid, BD3LM's published recipe ($$p_{\mathrm{MDLM}} = 0.85$$) lands roughly 3.7% worse in validation loss than the $$0.30$$ switch point at the same block size.

## Result 2: the optimum doesn't care about block size

The result that surprised me most isn't that 30% is good — it's that the *same* 30% is optimal at $$b = 4$$, $$16$$, and $$64$$, in both curriculum families. Naively, I expected the opposite: $$b = 4$$ is nearly autoregressive, so it should want lots of AR pretraining; $$b = 64$$ is much less AR-like, so it should want less. Instead the optimum stayed put.

<figure>
  <img src="/assets/images/block_length_invariance.png" alt="Best validation loss versus source fraction for AR-init and MDLM-init curricula across block sizes">
  <figcaption>The best switch point sits near 0.30 across block sizes up to 64, for both AR→BD3 and MDLM→BD3. Very large blocks (b=256) behave differently.</figcaption>
</figure>

My explanation: **the AR skills that transfer to diffusion are themselves block-size-independent, and they saturate early.** The AR phase is identical regardless of what $$b$$ you switch to, so whatever it teaches that BD3 can reuse — token statistics, syntax, semantics — is shared across all $$b$$. Past that point, additional AR training refines machinery specific to left-to-right prediction.

Here's a concrete toy example of why late-stage AR skills can be dead weight. Imagine training data containing strings like "$$A = p_1 \times p_2$$" where $$p_1 = 99{,}991$$ and $$p_2 = 100{,}003$$ are primes, so $$A = 9{,}999{,}399{,}973$$. An AR model that sees "$$A =$$" and must predict the right-hand side has to learn *integer factorization* — a genuinely hard mechanism. A BD3 model that sees "$$A = \_ \times p_2$$" only needs to *divide*. The two objectives reward fundamentally different computations, and AR's hard-won factorization circuit is useless under BD3 regardless of block size. Meanwhile BD3 has its own slow-converging skills (mask handling, bidirectional within-block reasoning) that AR never touches. The optimal switch point is set by AR-side dynamics — when the next AR step's *transferable* contribution drops below what a BD3 step would buy — and that quantity doesn't depend on $$b$$.

## The exception: half-sequence blocks

At $$b = 256$$ (half the sequence length), the curriculum collapses: every AR-init run blows past scratch immediately after the switch and never recovers, monotonically worse in $$p_{\mathrm{AR}}$$. Once the diffusion task is distributionally far enough from AR, the warm start stops being a warm start. Interestingly, MDLM-init partially survives at $$b = 256$$ — MDLM features are themselves diffusion-flavored, so some transfer remains even at large blocks.

<figure>
  <img src="/assets/images/loss_curves_b256_AC.png" alt="At block size 256, AR-init runs diverge from scratch after the switch; MDLM-init partially recovers">
  <figcaption>b=256. Left: AR-init runs blow past MDLM-init and scratch right after the switch. Right: best eval vs. source fraction — MDLM-init beats AR-init at every fraction, and only scratch is competitive.</figcaption>
</figure>

## The failure case was more interesting than I expected

dLMs are often motivated by data efficiency: when unique data is limited and you train for many epochs, diffusion keeps improving after AR starts overfitting. So the regime that actually motivates dLMs is exactly the regime where an AR warm start sounds risky. I tested the curriculum at unique-data pools $$U \in \{25\mathrm{M}, 50\mathrm{M}\}$$ tokens, 32 epochs each, $$b = 4$$:

| Method | $$U = 25\mathrm{M}$$ | $$U = 50\mathrm{M}$$ |
|---|---|---|
| Scratch BD3 (no WD) | **3.2079** | **2.9509** |
| Curriculum, naive ($$p_{\mathrm{AR}}=0.30$$, no WD) | 3.4884 | 3.1083 |
| Curriculum, tuned ($$p_{\mathrm{AR}}=0.30$$, AR-phase WD) | 3.2231 | 2.9603 |

The naive recipe fails badly — 0.16 to 0.28 nats behind scratch. Inspecting the runs showed why: the AR phase overfits early (at $$U = 25\mathrm{M}$$ its best eval comes at epoch 4, then degrades by ~0.4 nats before the handoff), and in parallel the MoE router collapses (final router entropy 0.747, 3% of tokens dropped). The BD3 continuation inherits a damaged router and never catches up.

The fix was *phase-specific regularization*: weight decay during the AR phase only (Muon WD 0.1 on matrices, small Adam WD, **no** router WD; all WD off during the BD3 phase). This flattens the AR phase, preserves the router (entropy 1.001, 0.04% dropped), and closes the gap to scratch from 0.28 to 0.015 nats at $$U = 25\mathrm{M}$$.

<figure>
  <img src="/assets/images/data_constrained_u25m_curves.png" alt="Data-constrained training curves comparing scratch BD3, naive curriculum, and tuned curriculum">
  <figcaption>U=25M, 32 epochs. The naive AR warm start (red) overfits and never recovers; adding AR-phase weight decay (green) makes the curriculum track scratch BD3 (black) closely.</figcaption>
</figure>

The lesson I keep coming back to: **the best AR checkpoint is not the best initialization.** At $$U = 50\mathrm{M}$$, the tuned AR phase had *worse* AR validation loss than the unregularized one (3.290 vs. 3.246) but produced a BD3 continuation 0.148 nats *better*. Router health, not AR loss, predicted the quality of the continuation. The representation you want to transfer is not the same thing as the source objective's best checkpoint.

## Caveats

I don't want to oversell a 70M-scale experiment, so the caveats up front. My budget is fixed in optimizer steps, not wall-clock: AR steps are roughly 5&times; cheaper than BD3 steps in my implementation, so a wall-clock-fixed comparison would favor a larger $$p_{\mathrm{AR}}$$. Everything is validation loss — at this scale downstream benchmarks are noise. The optimum almost certainly depends on scale, sequence length, and architecture (though as a downward data point, a 1M-parameter sweep on TinyShakespeare found the same 0.30 optimum). And MoE adds hyperparameter sensitivity that could shift absolute numbers.

**My current recipe:** when pretraining a small-block diffusion LM from scratch, spend ~30% of the budget on AR, then switch to BD3. If the data is multi-epoch, regularize the AR phase and watch the router. The exact number 30% is not a law of nature — but "convert a 97%-finished AR model" is not one either.
{: .notice--success}

## What I take away

Beyond the result, this project made me more excited about pretraining science. I like the combination of theory-shaped questions and very concrete engineering: optimizer state at the objective handoff, router entropy, weight-decay groups, loss curves that suddenly tell you something is wrong. And I like the discipline of compressing a question that sounds too large for my resources into one I can actually run.

A few months ago I did not know how to train these models. Over this project I learned to train MoE transformers, use Muon, implement diffusion objectives, and debug a curriculum that failed for a non-obvious reason. That is exactly the learning loop I want more of.

Next up: how this interacts with &mu;P and scaling methodology. If the point of pretraining science is making decisions predictable *before* spending the full budget, scaling rules aren't a side topic — they're the game.

---

*Full report: [A Block-Size Curriculum for Diffusion Language Model Pretraining](/assets/pdf/block-size-curriculum-dlm-report.pdf).*
