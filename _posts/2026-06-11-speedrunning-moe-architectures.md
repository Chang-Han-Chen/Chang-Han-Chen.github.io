---
title: "MoE autoresearch"
layout: post
date: 2026-06-11
excerpt: "My first attempt at autoresearch and scaling laws "
permalink: /blog/moe-speedrun-scaling/
---

*My first attempt at autoresearch and scaling laws*
{: .post-subtitle}

<details class="post-toc" markdown="1">
<summary>Table of Contents</summary>

* TOC
{:toc}
</details>

Based on the experience of [diffusion LM project](/blog/how-long-ar-before-diffusion/), I feel like I should do something more fundamental and principled. Looking back, the field has made tremendous progress on autoregressive GPTs, e.g. the architectures, the optimizers, the scaling methodology, the whole modern recipe, and working on the exotic frontier without having internalized that recipe felt like building on sand. It seems more reasonable to stand on the shoulders of giants first: get my foundations in LLM research as solid as possible, then carry them wherever I go next. So that's what this project is: architecture research on a plain autoregressive MoE, followed by an attempt to validate the findings the way the frontier labs do (I think), namely with scaling laws.

Tl;dr I didn't successfully get autoresearch to generate interesting ideas that work. With Karpathy's program.md Codex 5.5 xhigh basically just swept a finer grid of hyperparameters. I then iterated a few versions, got them to iterate on literature search and research, but still none of the ideas work eventually. Of course this could be due to the insufficiency of test time compute; I only let them work over one night max. In the end, I just did the research myself, asked it to implement techniques like XSA, gated attention, etc. One trick Codex found and did work is that under fixed-wall-clock constraint, first-two-dense-layer is helpful due to faster training. To see if these interventions work at scale, I wrote a plan for scaling laws, asked codex to implement and babysit the runs, and collected the results. Excitingly, the gain of our interventions look real at 10x the original scale and not diminishing! But of course we definitely need to go much beyond the $$\sim 4\times 10^{19}$$ FLOPs of our biggest run to be sure. 

For the impatient, here is the speedrun leaderboard — every intervention that became the new fixed-wall best (how each works: Appendix A; how they were found: Phase 1):

| # | intervention | val BPB | Δ vs prev | cumulative |
|---|---|---|---|---|
| 0 | clean simple backbone, LR 0.003 | 0.954881 | — | — |
| 1 | fixed value mix $$0.5 v_1 + 0.5 v_\ell$$ | 0.947597 | 0.763% | 0.763% |
| 2 | value mix $$0.75 v_1 + 0.25 v_\ell$$ | 0.945791 | 0.191% | 0.952% |
| 3 | sigmoid affinities + expert bias + load-balance 0.003 | 0.942848 | 0.311% | 1.260% |
| 4 | exclusive self-attention | 0.940518 | 0.247% | 1.504% |
| 5 | headwise attention gate, init 0.95 | 0.940352 | 0.018% | 1.522% |
| 6 | headwise attention gate, init 0.98 | 0.940076 | 0.029% | 1.550% |
| 7 | first FFN layer dense | 0.938510 | 0.167% | 1.714% |
| 8 | first two FFN layers dense | 0.938179 | 0.035% | **1.749%** |

A bunch of disclaimers upfront: our MoE router might be suboptimally tuned, but I put significant effort into making it at least look "healthy"; our scaling laws have fewer data points than I'd like; everything is single-seed; no layer-specific hyperparameters (so the embedding layer is likely undertrained) like &mu;P, but we did sweep global learning rate at each scale. 

Still, I'm writing it up anyway. This is a learning process, and hopefully by exposing the process (mistakes included), I can collect on-policy, dense reward signals from experts out there. Let me know if you have any feedback!

The project has two halves, and so does this post:

1. **A speedrun.** Karpathy's autoresearch style. Starting from a deliberately stripped-down MoE transformer, Codex ran dozens of architecture experiments under a fixed 7.5-minute training budget on 4 H100 GPUs, following a protocol document I wrote. We kept a stack of 8 ideas, most of which came from myself though, improving validation bits-per-byte by 1.75%.
2. **A scaling check.** We then asked the question that toy-scale architecture work usually dodges: do these wins survive when you scale the model up and train each size to a compute-optimal token budget, with paired controls?

**Question: are architecture wins from a 7.5-minute speedrun real? Do they persist under a compute-optimal scaling?**
{: .notice--info}

The answer, at the scale I could afford: **mostly yes**. The full intervention stack beats the simple backbone at 76M active parameters (by 1.49% relative BPB) wins clearly at 185M (by 2.72%), but ties at 91M. So, the gap is scale-positive but non-monotone. My observation is that most of the gain may come from the **stability** of training. For the baseline architecture, we saw repeated loss spikes and ended with visibly unhealthy routing even at 185M-active scale, while the recipe with our interventions stayed clean at every size up to 565M active parameters. A late-arriving ablation then rearranged my whole reading of the curve: most of the non-monotonicity traces to *one* intervention (the dense stem), and removing it both fixes the awkward middle point *and* exposes the stability problem that motivates what I want to do next.

## Phase 1: speedrun

### Setup

Everything runs on [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) harness, lightly adapted: a single `train.py` (the only file the agent may modify), a frozen `prepare.py` that downloads data and defines the benchmark, and two log files the agent must maintain. Data is [ClimbMix](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle), tokenized with a custom 8192-vocab BPE, sequence length 2048. Hardware is a 4×H100 node rented from VESSL AI and Vast.ai. As promised in my [first post](/blog/a-new-chapter/), this kind of research really is *doable* on a PhD student budget (but very tight😱).

The benchmark metric is **validation bits-per-byte (BPB)**, following Karpathy: per-token cross-entropy summed over the *bytes* of the target text, rather than averaged over tokens. This makes models with different tokenizers comparable and prevents an architecture from "winning" by exploiting token granularity. Lower is better; at this scale, differences of ~0.001 BPB are meaningful and reproducible, which still surprises me.

The model family is a **mixture-of-experts (MoE) transformer**. Quick reminder on MoE: instead of every token passing through the same feed-forward network (FFN), each token is *routed* to a small subset of parallel FFNs ("experts"), so the model has far more total parameters than it spends compute on per token. In our case, each forward pass picks the top 2 of 16 experts, and the model has 438M *total* but only 91M *active* parameters. The catch is the **router**, a tiny linear layer that decides which experts see which tokens. It is notoriously hard to stably train the routers; a common instability is that a few experts happen to get routed more, get more gradient signals, perform better, and therefore get routed even more often. So, MoE experiments must track router health (load balance, entropy, per-expert load) alongside the loss. In my own experiments, many architecture interventions for dense transformers don't work for MoE because they somehow mess up the router; for example, learnable value residual.

The speedrun protocol: each run trains for a fixed **450 seconds of measured wall-clock training time**, then evaluates validation BPB. Fixed *time*, not fixed steps. So an intervention can win either by learning more per step or by being faster. This feels like a more practical constraint, especially for MoE, because their architecture (e.g. sparsity) can often greatly impact hardware efficiency (MFU, ...).

### My program.md

Here is the workflow, which is the part of this project I find most interesting. I do not sit at the terminal supervising runs. Instead I write a `program.md` — a protocol document — and Codex (5.5 xhigh, for everything reported here) reads it, edits `train.py`, launches runs, parses the logs, and appends structured entries to a `research_log.md`. The program is a contract: it specifies what the agent may and may not touch, how to decide keep-vs-discard, and — crucially — the *epistemic* discipline. From the architecture program:

> The key discipline is: **pre-register the hypothesis before launching the run, then update the belief after reading the result**. Do not simply try changes because they are fashionable or because they appeared in a speedrun.

Every run entry must contain a pre-run hypothesis (a mechanism, not a vibe), an expected result *including at least one predicted diagnostic besides BPB*, the observed result, an interpretation, an explicit `agrees with hypothesis: yes/no/partial`, and a decision. My favorite line in the program — the kind of sentence you only write *after* watching an agent burn GPU-hours on undirected fiddling:

> A run without an interpretation is not a scientific run; it is just a CUDA-powered coin toss.

The program also hard-codes guardrails: the optimizer family is frozen (Muon for matrices, AdamW for the rest — only LR *values* may change), `prepare.py` and the metric are untouchable, and router-health thresholds define when a "win" must be marked `repair` instead of `keep` (e.g., any layer where one expert receives ≥25% of top-2 traffic over 16 experts).

I want to be honest about the division of labor, because I think it's the most useful data point in this post for anyone considering this workflow.

Codex did not need help with execution. It wrote correct grouped-matmul MoE dispatch code, managed git hygiene, parsed its own logs, aborted bad runs early against pre-registered criteria, and maintained the research log with a discipline I frankly envy. What it needed constant help with was **research taste**. With Karpathy's reference `program.md` essentially as-is, Codex just swept a finer and finer hyperparameter grid — each run a locally sensible "one more bracket point," and collectively a pile of compute spent polishing the fourth decimal of a learning rate. Nothing in its behavior was wrong, exactly. It just wasn't *research*. I rewrote the program over several iterations to push exploration: explicit intervention queues, a coarse-only 3× LR grid rule, instructions like "do not continue this thread unless the result creates a new clear failure mode."

The most ambitious version of the program hands Codex the idea generation itself, with structure: run 20 search/reflection iterations, each scanning ~10 papers, selecting 2 one-sentence ideas with a written reason why those 2 beat the other 8, then updating the search keywords for the next iteration; afterwards, distill the 40 collected ideas through timed synthesis passes into a unified hypothesis and a ranked experiment queue, under a research filter the program calls "brutal: prefer ideas that can be stated in two sentences, implemented in a few lines, and tested with one clear diagnostic besides BPB." And I have to say, the output *looks* like research. The reading list is diverse and feels high-signal — selective token losses (Rho-1 style), multi-token and ranking objectives, JEPA-style latent targets, curriculum and data-ordering work — the queue is sensibly prioritized by implementation cost, and it even surfaced a genuine repo-specific insight on its own: we train mean token CE while evaluating byte-weighted BPB, so the training objective and the benchmark metric are misaligned by construction. But when its ideas were actually tried, none of them produced a keep. My current guess is test-time compute: I gave it one night max per attempt, which is enough to scan and rank a literature but probably not enough to iterate on any single idea until the details that make a paper's trick work are faithfully reproduced. I'd honestly like to be wrong here, and I want to rerun this loop with a much bigger inference budget someday.

So in the end I did the idea generation myself, from paper reading. Of the five interventions in the final stack, I proposed four — the value-mix family, the sigmoid/bias router, exclusive self-attention, and the headwise attention gate all entered through the program's suggested queue. **Exactly one adopted idea originated with Codex: making the first FFN layers dense.** It's a good trick, mainly because it trains faster under the fixed wall clock (more on the quality side later — the scaling phase has things to say about it). DeepSeek-V3 does the same thing with three dense layers; Codex found two was optimal at this scale, and tested three, which lost. But the scorecard matters for calibrating claims about "AI doing AI research": in this project, Codex was a tireless, rigorous research *executor* and run babysitter, not a research *director*.

One more honesty note: the research log I'm quoting was **reset** partway through the project. The first lineage of experiments accumulated a feature-rich stack (value embeddings, logit softcaps, …) whose components had never been individually validated, and when I grew suspicious, we stripped everything back to a minimal backbone and re-justified each component from scratch. Everything below is from the clean second lineage. The first lineage survives only in git history and in my humility.

### The speedrun

The starting point is deliberately boring: an 8-layer, 768-dim transformer, 16 experts with top-2 routing and softmax token-choice, GQA + RoPE + QK-norm, no value tricks, no gates. The first three runs just bracket the AdamW peak LR on a 3× grid — `0.01` was too hot (router load badly skewed), `0.001` undertrained catastrophically (BPB 1.031!), and `0.003` became the clean baseline at **0.954881 BPB**. Worth pausing on that middle number: at fixed wall-clock, a 3× LR error costs *vastly* more than any architecture intervention we found. Architecture research at fixed budget is conditional on getting LR roughly right, which is why the program forces an LR re-check whenever model scale changes.

From there, Codex worked through the intervention queue. The leaderboard at the top of the post lists every run that became the new best; rather than narrate all ~30 runs, here is the progression at a glance and then the three stories I think teach the most.

<figure>
  <img src="/assets/images/speedrun_progression.png" alt="Phase 1 fixed-wall leaderboard progression from 0.9549 to 0.9382 BPB across eight interventions">
  <figcaption>The speedrun staircase. Eight successive fixed-wall bests, cumulative −1.75% BPB. Not shown: the much larger number of runs that died on this hill. Math and code for each intervention are in Appendix A.</figcaption>
</figure>

#### Story 1: the value residual that failed, and the diagnostic that explained why

The first idea I queued was a [ResFormer](https://arxiv.org/abs/2410.17897)-style value residual: let later attention layers mix in the *first* layer's value tensor through a learned scalar per layer, initialized at zero. In dense transformers this reliably helps. Here, it didn't — matched-step loss was no better and router load concentrated. The interesting part is what the agent did next. Instead of shrugging and moving on, it added gradient instrumentation and ran a 120-second diagnostic with retained activation gradients, then wrote what I consider the best paragraph in the entire log:

> The scalar residual is not inert: mean absolute `value_resid_alpha` grew from 0 to 0.225 by step 596, with one layer reaching 0.968. However, the residual path gradient into the retained first-layer value clone stayed tiny in absolute RMS […] The data argues against the "useful early value gradient highway" story for this exact form. The scalar learns a sizable residual mixture, but the visible effect is mostly representation/routing disturbance.

In other words: the mechanism story from the dense-model literature ("value residuals create a gradient highway to early layers") measurably did not transfer. The learned mixture was *changing the representations that later routers see*, destabilizing expert load before delivering any gradient-flow benefit — a failure mode that simply cannot exist in a dense model, because dense models have no router to disturb. The repair, two runs later, was almost comically blunt: don't learn the mixture at all. A **fixed** mix $$v = 0.75\,v_1 + 0.25\,v_\ell$$ — first-heavy, damped, with nothing for the optimizer to over-adapt — turned the idea from harmful to a 0.95% cumulative win. Learned, normalized-learned, and larger-ratio variants were all tested; all lost. I would not have guessed that the difference between "harmful" and "best win of the phase" was *removing* the learnable parameter.

#### Story 2: the router that collapsed on schedule, and the control that earned the win

The router thread is my favorite example of the pre-registration discipline paying off. The target was DeepSeek-V3-style routing ([sigmoid affinities](https://arxiv.org/abs/2412.19437) plus [auxiliary-loss-free expert bias](https://arxiv.org/abs/2408.15664)): score experts with independent sigmoids instead of a competitive softmax, and steer load not with a loss but with a non-gradient bias term that nudges selection toward underused experts.

Tested with zero load-balance loss — the purist version — it improved BPB and then did exactly the bad thing:

> However, one layer collapsed: max-layer load CV 2.646, max-layer max load 0.500, min-layer router entropy 0.000884, and max router bias hit the 0.25 clamp.

One layer's router sent half of all top-2 traffic to a single expert of sixteen. Here's the part I'm proud of (as the program author): the *repair was pre-registered before the failure occurred* — the program said that if loss-free balancing collapsed, the agent got exactly one run adding a small differentiable load-balance term, coefficient 0.003, no sweeping. That run fixed the collapse (max-layer max load 0.0746, basically uniform) *and* improved BPB further, to 0.942848.

Then the step most tempting to skip: Codex ran a **control** — same value mix, same small 0.003 coefficient, but plain softmax routing — to test whether the win was really the sigmoid/bias mechanism or just the lighter load-balance pressure (the baseline coefficient had been 0.0085). The control regressed badly to 0.947950 with terrible load (max-layer max load 0.485):

> Lowering the load-balance coefficient is not sufficient. Without sigmoid/bias, softmax routing becomes highly imbalanced […] The best run's improvement really depends on the sigmoid/bias router controller.

Without this control, the honest claim would have been "we changed two things and it got better." With it, the mechanism is isolated. This is the cheapest kind of rigor there is, and I had to put it in the program explicitly because nothing about a leaderboard rewards it.

#### Story 3: Codex's own idea, and the confound it had to defuse

After the attention-side wins (exclusive self-attention, then a near-identity headwise sigmoid gate — see Appendix A; the gate-init bracketing 0.95 → 0.98 → 0.99 is a tidy little study in itself, converging on "as close to identity as possible but not closer"), the agent proposed the one original idea that survived: **replace the first MoE FFN layers with plain dense SwiGLU layers** of matched active width. The hypothesis, in its words: early token representations may be "too raw for useful sparse routing," so spend the first layers on a dense transformation and save the experts for later layers where routing decisions are better-informed.

One dense layer: 0.938510. Two dense layers: 0.938179, with total parameters *down* from 554M to 438M (each dense layer replaces a 16-expert bank) and throughput *up* (24.6% vs 22.3% MFU). Three dense layers: worse — the benefit saturates fast.

But notice the trap. Under a fixed wall clock, the two-dense model is faster, so it completed 2778 steps where the previous best completed 2515. Is the BPB gain architecture, or just more steps? The fixed-wall protocol *cannot answer this* — its two virtues (rewarding quality and rewarding speed) are confounded by construction. So the agent ran the two-dense model again, stopping at exactly 2515 steps:

> `val_bpb 0.939195` at exactly 2515 steps […] This beats the no-dense attention-gate baseline at the same step count (0.940076) by 0.000881 BPB […] The two-dense gain is not purely an artifact of getting more updates in the 450s budget.

So the dense stem is doing two things at once — a small genuine sample-efficiency gain (0.094% at matched steps) and a larger throughput gain (the rest of the fixed-wall delta). I like this run because it converts an ambiguous leaderboard entry into two cleanly separated claims. And as noted above, it independently rediscovers actual frontier practice: DeepSeek-V3 keeps its first three FFN layers dense, reportedly for exactly this early-routing-stability reason. (Hold this thought, though — the scaling phase has a plot twist waiting for the dense stem.)

#### The failure museum

The log contains far more discarded runs than kept ones, and some failures were expensive. The most instructive: I asked for depth-scaled residual scaling (multiply layer-$$\ell$$ branches by $$1/\sqrt{\ell}$$, a trick with decent dense-model pedigree), and **eight consecutive runs** tried every placement — pre-norm, branch-only, MLP-only quarter-power (a cute hypothesis: SwiGLU is roughly quadratic in input scale, so scale inputs by $$\ell^{-1/4}$$ to get an $$\ell^{-1/2}$$ branch), MLP+V, MLP+V+gate, V-only, V+gate, and finally output-side. Every variant lost. The post-mortem ordering is at least informative — internal scaling was catastrophic, output-side scaling merely bad, with a consistent moral: this model wants full-amplitude branches — and the family was finally closed with the log's dry "Close fixed deterministic layer scaling for now." In retrospect the family should have been capped at three runs in the program. And I have to own my half of it: reading the log carefully, at least two of the eight variants were *my* mid-flight requests, not the agent's ("iterate once more by also scaling the headwise attention gate as requested"). The sunk-cost gradient of locally-reasonable next-variants pulls on humans too; the difference is the agent writes it all down.

Also in the museum: fine-grained experts (32 experts, top-4, half-width — preserves active FFN width, total expert width, and sparsity ratio), which DeepSeek-style scaling folklore says should help. It lost decisively *three times* — once early, once re-tested on top of the stabilized router stack in case the old router had been the bottleneck, and once more in the formal Phase 2 geometry check (0.953126 vs 0.935178, not close). Routing was perfectly healthy each time; the smaller experts just learned worse and dispatched slower at this scale and horizon. I file this under "folklore is scale-dependent" and would genuinely like to know where the crossover is.

## Phase 2: scaling laws

This is the part of the project I was most excited about, and the part where I made my biggest methodological mistake. Phase 2 got its own program, `program_june_ninth.md` — much more boring to read than the first, by design, since its job was specification rather than exploration. It specifies three things: pick the MoE geometry to scale, scale the recipe across five sizes (F1–F5: depth 8→16, width 768→1792, 91M→852M active params) with paired controls, and decide the outcome by **pre-registered verdict definitions**, fixed before any scale run launched:

> Call the architecture interventions "scaling" if `simple_minus_full` is positive at every completed depth, and the relative BPB reduction is flat or increasing, or only mildly decreasing. Call the gains "diminishing" if `simple_minus_full` shrinks monotonically with depth, or the depth-16 relative reduction is less than half the depth-8 one. Call the result "scale-dependent retuning needed" if full beats simple at small depth but loses at large depth, [or] router-safe status changes with size […]

The "simple" curve is the paired control: identical depth, width, expert geometry, and optimizer, with all five interventions switched off and the baseline softmax router restored. Every claim below is full-vs-simple at matched scale, not "bigger model got better."

### Picking the geometry to scale

First, expert count at fixed top-2 and fixed active compute, E ∈ {4, 8, 16, 32, 64, 128}:

<figure>
  <img src="/assets/images/sparsity_sweep.png" alt="Sparsity sweep: validation BPB is U-shaped in expert count with minimum at E=16; steps completed in fixed wall time decrease monotonically with expert count">
  <figcaption>Fixed-wall sparsity sweep at 91M active params. Left: BPB is U-shaped in expert count — capacity helps until optimizer state and memory traffic eat the budget. Right: why — a 128-expert model (3.2B total params!) completes 2.6× fewer steps than a 4-expert model in the same 450 seconds. E=16 (12.5% sparsity) wins and was selected.</figcaption>
</figure>

The shape is intuitive in hindsight: under fixed wall-clock, inactive parameters are not free — they cost optimizer state, memory traffic, and dispatch overhead, so E=64 and E=128 train stably but are hopelessly outrun. What I appreciate is the discipline at the bottom of the U: E=8 trails E=16 by only 0.003 BPB while being smaller and faster, and a pre-registered tie/scale-candidate rule existed to handle exactly this. (It didn't trigger — no contender came within 0.001 — but deciding the rule before seeing the data is the point.) E=16, top-2, 12.5% sparsity it is.

### The mistake: fixed-wall scaling is not a scaling protocol

My first instinct — embarrassing in retrospect — was to just run the bigger models under the same 450-second wall clock. The F2 model (185M active) reached 1434 steps and a BPB of 0.973, *worse than the 91M model*. F3 (352M active) reached 751 steps, BPB 1.046, and crossed a router safety threshold for the first time in the entire project. These numbers say nothing about architecture: a fixed time budget gives every model the same seconds, so larger models are simply measured earlier in their training curves. Within the protocol we tried one LR rescue at F3 (lower LR fixed the router warning and made BPB *worse* — the problem was undertraining, not instability), which at least diagnosed the issue cleanly.

The fix was to switch protocols mid-project: train each size to a **compute-optimal token budget** of ~20 tokens per active parameter (the Chinchilla rule of thumb), on the full 400-shard data pool (~25.3B tokens — even the biggest run stays under one epoch), with the cosine schedule horizon matched to the actual step budget, and a coarse 3× LR grid (`0.000333 / 0.001 / 0.003`) re-swept at each size. So F1 trains on 1.8B tokens, F2 on 3.7B, F3 on 7.0B, F4 on 11.3B — runs now take hours, not minutes. This is the obvious-in-hindsight standard protocol; I just hadn't internalized *why* it's standard until I watched the fixed-wall version produce nonsense. Learning by generating the failure yourself is expensive but very durable.

One detail I now appreciate: the LR re-sweep is not optional. The optimal AdamW peak LR moved from 0.003 at 91M (where 0.001 loses by a brutal 0.044 BPB) to 0.001 at 185M and above (where 0.003 loses, and at F3 a matched-step diagnostic showed it falling behind *and* destabilizing the router within the first 10% of training, so we aborted rather than spend the full budget — the coarse grid plus early-stop diagnostics kept the sweep affordable). If we had pinned LR at either value across all scales, the scaling curve would be silently corrupted at one end. I knew "LR doesn't transfer across scale" as a slogan (this is what μP is for!); there's a difference between knowing the slogan and watching 0.044 BPB evaporate.

### The result

| size | active params | tokens | full BPB | simple BPB | gap (rel.) |
|---|---|---|---|---|---|
| S75 | 76M | 1.5B | 0.902337 | 0.916017 | **1.49%** |
| F1 | 91M | 1.8B | 0.892152 | 0.892165 | **0.00%** (tie) |
| F2 | 185M | 3.7B | 0.821625 | 0.844628 | **2.72%** |
| F3 | 352M | 7.0B | 0.765974 | — | — |
| F4 | 565M | 11.3B | 0.736565 | — | — |

<figure>
  <img src="/assets/images/scaling_full_vs_simple.png" alt="Left: compute-optimal validation BPB vs active parameters for the full recipe (S75 through F4), the simple control (S75 through F2), and the all-MoE ablation (S75 through F2, with a router warning marker at F1). Right: grouped bars of relative BPB reduction vs the simple control — full recipe: 1.49%, tie, 2.72%; all-MoE: 1.83%, 1.87%, 2.36%">
  <figcaption>Left: the compute-optimal curves. The full recipe scales smoothly to 565M active params with healthy routing throughout; the dashed all-MoE ablation (next section) is the same recipe minus the dense stem. Right: the architecture gap vs the simple control. The full recipe's gap is scale-positive but non-monotone, with a dead tie at F1; the all-MoE gap is consistent at every size. The non-monotonicity was the dense stem's doing.</figcaption>
</figure>

Three observations, in decreasing order of confidence.

**First, the stack scales mechanically and stays healthy.** Up to 565M active (3.3B total) parameters, no divergences, no router collapse, max-layer expert load near-uniform at every size. That sounds like faint praise but it isn't: the simple control at F2 suffered repeated transient loss spikes above 6 during training and finished with a mean load CV of 0.52 against the full recipe's 0.075. Part of what the intervention stack buys is apparently not lower loss but *lower variance and router sanity at scale* — which at real scale is worth more than the BPB, since a diverged run returns nothing.

**Second, the gap is scale-positive but non-monotone, and the F1 tie is real.** At F1 the two recipes land within 0.000013 BPB of each other — that is a tie to four decimal places, at the *exact* configuration where the entire Phase 1 speedrun was conducted. I found this genuinely confusing and a little funny: the size where we did all the architecture search shows no compute-optimal-matched advantage at all, and the wins only separate at the scales we never searched at. (The ablation in the next section finally produced a suspect.) The log is appropriately blunt about not overclaiming it:

> F1 remains the weakest point: the direct validation comparison is a tie, so any fixed-wall advantage there is mainly a throughput/sample-count effect and should not be overclaimed without validating a simple checkpoint near the cutoff.

(Note what the tie does *not* say: at fixed wall-clock F1-full still wins, because the full recipe is faster — the dense stem's throughput gain is real at every size. The tie says that at matched tokens, the quality side of the F1 advantage is zero.) One consistency check against the F2 win being an artifact of LR selection: the full recipe beats the simple control at F2 even at the *same* LR 0.003 (0.822997 vs 0.844628, a 2.56% gap), so the architecture gap there is not "full got a better-tuned LR."

**Third — and this is the asterisk on everything — the scale-positive claim rests on F2.** With controls at only three sizes and the middle one a tie, the honest summary of the gap curve is "two wins, one tie, wide error bars of unknown width." Everything is one seed. By the program's own pre-registered definitions, the verdict so far is "scaling" (the gap is positive-or-zero everywhere and not shrinking), but the definitions assumed five control points and I have three. The single highest-value experiment I haven't run is the **F3 simple control** at 7B tokens: if the gap holds or grows at 352M, the claim gets real legs; if it ties again, the F2 point starts looking like the outlier and the whole story inverts. It's queued. (F5, at 852M active and 17B tokens, is also unrun — it stopped fitting my budget. And before the LR-rescue at F4: the 0.0003 arm was aborted at 79% of budget, clearly undertrained at matched windows, so F4's 0.001 number stands unbracketed from below. Imperfections, as promised.)

### The twist: ablating Codex's favorite intervention

The June 9 program defined one more control I almost didn't run:

> **Optional Control: No-Dense Stabilized Stack.** Run this only if the full-vs-simple delta is confusing or if we need to isolate the dense-stem contribution.

The full-vs-simple delta *was* confusing — the F1 tie — so after the main sweep I ran the **all-MoE ablation**: the complete intervention stack (value mix, sigmoid/bias router, XSA, headwise gate) with `DENSE_EARLY_LAYERS=0`, every FFN layer sparse, at each size's selected LR anchor. I expected it to confirm the dense stem's importance. It did the opposite:

| size | all-MoE BPB | full (two-dense) BPB | all-MoE − full | router health |
|---|---|---|---|---|
| S75 | **0.899251** | 0.902337 | **−0.0031** | clean (max-layer max load 0.073) |
| F1 | **0.875468** | 0.892152 | **−0.0167** | ⚠ one layer: max load **0.253**, CV **0.865** |
| F2 | 0.824722 | **0.821625** | +0.0031 | clean (max-layer max load 0.073) |

Two things happened at once here, and they pull in opposite directions.

**The good news: the F1 mystery mostly dissolves, and the quality story gets cleaner.** All-MoE *beats* the full recipe at S75 and crushes it at F1 — 0.875 vs 0.892 is a 1.87% gap, enormous by this project's standards. Recompute the architecture gap against the simple control using all-MoE instead of the two-dense recipe and the embarrassing non-monotonicity nearly vanishes: 1.83% at S75, 1.87% at F1, 2.36% at F2 (right panel of the figure above). In other words, *the intervention bundle's quality gain was consistent at every scale all along* — and the F1 tie was substantially the dense stem giving that gain back. Under the speedrun's fixed-wall protocol the stem's throughput advantage masked this completely; under compute-optimal token budgets, where speed buys nothing, the first two dense layers are revealed as a quality *cost* at small scale. The intervention I praised three sections ago as the agent's best idea is, at compute-optimal S75 and F1, a net negative that was hiding inside a benchmark artifact. I keep rereading Story 3 and it holds up — the matched-step diagnostic was real — but matched-step-at-fixed-wall-horizon and compute-optimal-matched are different questions, and I only learned to distinguish them by getting burned here.

**The bad news: we can't actually have the all-MoE model yet.** Look at the F1 router column: one layer finished with max expert load 0.253 — across the pre-registered 0.25 safety threshold, max-layer load CV 0.865. Every other compute-optimal run with the full intervention stack, at every size up to 565M, stayed comfortably clean; the first time we remove the dense stem, the safety threshold gets crossed — attached, of course, to the best F1 number in the project. The log refuses to celebrate, correctly:

> The F1 all-MoE result is a large validation win but has one imbalanced router layer, so treat it as a high-signal result that deserves a repeat or layer-level diagnostic before making it the default.

And at F2, all-MoE flips to losing: 0.0031 BPB worse *and* meaningfully slower (23.5% vs 25.7% MFU — without the dense stem, every layer pays expert-dispatch overhead). The pattern across the three sizes is uncomfortable: at LR 0.003 the all-MoE model develops a hot layer as it grows; at F2's gentler LR 0.001 it stays clean but can no longer keep up. Either you flirt with instability or you pay a tax to avoid it. The dense stem, in this light, was never really a quality intervention — it was a *stabilizer* whose price increases with the compute budget, which is presumably why DeepSeek-V3 (trained at fixed token budgets, not fixed wall-clock) still ships dense early layers: at their scale the insurance is worth it.

So the model I actually want — fully sparse, no dense crutch, stable at every size — is sitting right there in the table, one imbalanced layer away. The two-dense recipe remains the default for F2+ scaling for now, per the log. But "for now" is doing a lot of work in that sentence, and the next section of my research life is about removing it.

## Takeaway

What did the speedrun actually buy? My attempt at an honest decomposition, post-ablation. The interventions split into two groups with different jobs. The **four attention/router interventions** (value mix, sigmoid/bias routing, XSA, headwise gate) bought *quality*: a consistent 1.8–2.4% BPB gap over the simple backbone at every compute-optimal size tested, plus spike-free training and router sanity at sizes where the control gets ugly — and I'm fairly confident in this number precisely because it's *boringly* consistent across scales. The **dense stem** bought *throughput and insurance*: an unambiguous speed advantage at every scale, and protection against the layerwise router imbalance that appears the moment you remove it — at the price of giving back quality at small scale (all of it at F1, where it single-handedly produced the tie) and a smaller net positive at F2. The fixed-wall leaderboard number that drove the whole speedrun (−1.75%) was a *mixture* of quality, speed, and stability. A fixed-wall benchmark is a great search signal precisely because it rewards that mixture — and a bad reporting metric for the same reason. Search on fixed-wall, claim on compute-optimal-matched, and ablate component-by-component before telling yourself a story about *why* the stack wins.

On methodology: I came into this wanting to learn scaling-law practice, and the way I actually learned it was by doing it wrong once at full speed. Fixed-wall scaling produced confident nonsense, and the protocol pivot — compute-optimal budgets, paired controls, per-scale LR re-sweeps — was forced on me by my own garbage data rather than absorbed from a paper. Then the all-MoE ablation taught the same lesson one level deeper: even after the protocol was fixed, my *interpretation* ("Codex's dense stem is the star") was an artifact of the search protocol, and only a component ablation under the claim protocol could correct it. I suspect this is the only way these lessons really stick.

**Recipe, if you want to try this:** write the program before the experiments; pre-register hypotheses *and* repair rules *and* verdict definitions; force one diagnostic beyond the headline metric on every run; make controls non-optional in the document (the agent will not volunteer them); cap exploration depth per idea family; re-sweep LR coarsely at every scale change; and budget for the paired-control curve from the start — it is the difference between a scaling claim and a scaling anecdote.
{: .notice--success}

### Working with a research agent

The subtitle of this post is the one-line readout: I couldn't get Codex to come up with ingenious ideas yet, but it is very useful for babysitting runs.

**Great at** (and I mean better-than-me-at-2am great): execution fidelity, log discipline, and *negative-result hygiene*. It never quietly dropped a failed run, always wrote down matched-step comparisons with exact numbers, aborted runs against pre-registered criteria instead of hope, and once spent a diagnostic run instrumenting gradients to figure out *why* an idea failed rather than just recording that it did (Story 1). A surprising fraction of human research pathology is absent by default. This mattered most in the scaling phase: multi-hour runs with LR sweeps and abort criteria are exactly the kind of tedious, vigilance-heavy work where a tireless babysitter beats a sleepy human.

**Bad at:** knowing which question is worth asking next. Its failure mode is never laziness — it's a kind of diligent myopia, generating an endless sequence of locally-justified next steps down whatever gradient is in front of it (finer LR grids, an eighth depth-scaling variant). And when I explicitly asked for creativity via the literature-search program, I got ideas that *sounded* like research — diverse, well-motivated, properly cited — and didn't survive the benchmark, plausibly for lack of iteration compute. The program is the steering mechanism, and the program is where the research taste lives. I rewrote it many times, and each rewrite was prompted by watching a specific unproductive behavior and asking "what sentence would have prevented this?" (Standard caveats: this is one project, one model, at most one night of test-time compute per attempt. I fully expect this readout to age.)

**Didn't expect:** writing the program made *me* a better experimentalist. Pre-registering verdict definitions, separating search metrics from claim metrics, deciding repair budgets before failures happen — none of this is agent-specific. It's just methodology that becomes *mandatory* when your collaborator executes everything literally and never pushes back. The agent is a mirror with a very high frame rate. (My PhD advisor will be pleased that "lower your standards, write down the boring calculation explicitly" turns out to be transferable from quantum gravity to MoE routers.)

## Final thoughts

This post is not a conclusion; it's just a beginning. The all-MoE ablation handed me both a gift and a bill: the gift is a cleaner story (the intervention bundle scales; the stem was insurance), the bill is a concrete open problem — **train the fully sparse model stably at every scale, without buying insurance from dense layers**. I have two leads I want to chase. One is the router itself: the current load-balance machinery (mean-statistics losses plus a bias controller) demonstrably lets a single layer drift toward imbalance while every *average* looks immaculate, and I want to study quantile balancing — QB — which targets the tail of the expert-load distribution rather than its mean, which is exactly where my failures live. The other is principled scale transfer: I tuned LR by brute-force sweep at every size in this project and watched the optimum move under my feet (and per the disclaimers, my layer-agnostic global LR likely leaves the embedding layer undertrained); μP is the tool that's supposed to fix both, and I now have very personal motivation to understand when it actually works.

But, well, I clearly set too high a goal again. The lesson of my PhD was to lower my standards, validate the simplest thing I can fully understand, and iterate; the lesson of this project is that I still haven't fully internalized it. "Stabilize all-MoE at every scale" is not the simplest thing I can fully understand — I can't even claim to fully understand how to stabilize and scale a *dense* model yet, with hyperparameters that transfer instead of hyperparameters re-bracketed by brute force at every size. So, lower the standards once more: dense models first. That's μP, and that's the next post. Then, with that foundation, back to MoE: implement QB, point it at the F1 all-MoE hot layer, and see whether that is enough to stabilize the runs and let the dense crutch go.

(The algorithm phase is also queued — its program is the literature-search one from earlier, the one whose ideas haven't earned a keep yet; its top-priority item is fixing that train-CE-vs-BPB misalignment Codex spotted, and I want to rerun the whole idea-generation loop with far more inference budget. So many threads, so few GPUs. It feels great to be at the beginning of something.)

---

# Appendix

## A: the kept interventions, in math and code

All code below is the actual `train.py` implementation (lightly trimmed). Notation: $$x \in \mathbb{R}^{B\times T\times d}$$ is the residual stream, $$h$$ heads, $$E=16$$ experts, $$K=2$$ active.

### A.1 Fixed first-value mix

**Idea.** Layers $$\ell \geq 1$$ compute attention not over their own value vectors $$v_\ell$$ but over a fixed mixture with the *first* layer's values:

$$v^{\text{mix}}_\ell = \alpha\, v_1 + \beta\, v_\ell, \qquad \alpha=0.75,\ \beta=0.25.$$

This descends from value-residual learning ([ResFormer](https://arxiv.org/abs/2410.17897)): early value streams carry token-local information that later layers benefit from re-accessing directly. The MoE-specific twist, learned the hard way (Story 1), is that the coefficients must be **fixed, first-heavy, and damped**: a learned scalar version measurably disturbed the representations seen by downstream routers and concentrated expert load before delivering any benefit. Removing the learnable degrees of freedom is what made the idea work. Note $$\alpha+\beta=1$$, so the mixed value's scale stays comparable to a plain value vector.

```python
# CausalSelfAttention.forward — v is this layer's values, first_v is layer 0's
layer_v = v   # cached so layer 0 can hand its values to later layers
if self.value_mix_enabled and first_v is not None:
    mixed_v = (self.value_mix_first.to(v.dtype) * first_v.to(v.dtype)
               + self.value_mix_local.to(v.dtype) * v)
    v = mixed_v
```

### A.2 Sigmoid-affinity router with loss-free expert bias

**Idea** (from [DeepSeek-V3](https://arxiv.org/abs/2412.19437) / [aux-loss-free balancing](https://arxiv.org/abs/2408.15664)). Two coupled changes to top-k token-choice routing.

*Scoring.* Replace softmax routing probabilities with independent sigmoid affinities. With router logits $$z = W_r x \in \mathbb{R}^E$$:

$$a_i = \sigma(z_i), \qquad \text{output weight } w_i = \frac{a_i}{\sum_{j \in \text{top-}k} a_j} \ \text{ over selected experts only.}$$

Softmax makes experts compete globally — pushing up one expert's score necessarily pushes down the others' — and this competition is entangled with the load-balancing gradients. Sigmoid scores each expert independently and normalizes only at the end, over the selected experts.

*Load control.* Instead of relying on an auxiliary loss, keep a non-gradient per-expert bias $$b$$, updated from an EMA of realized hard load: experts above uniform load get pushed down, below get pushed up. Crucially, the bias affects **selection only** — output weights use the clean affinities, so the correction never contaminates the forward computation:

$$\text{select top-}k \text{ of } (a + b), \qquad \text{mix with } w(a) \text{ only}.$$

$$b \leftarrow \mathrm{clamp}\!\left(b + \eta\, \frac{u - \mathrm{EMA}(\text{load})}{u},\ \pm 0.25\right), \qquad u = 1/E.$$

*The empirical footnote that matters:* pure loss-free balancing collapsed one layer at this scale (Story 2). The working configuration keeps a small differentiable load-balance term, coefficient 0.003 — about 3× lighter than the softmax baseline needed — and the softmax control proved the mechanism is the sigmoid/bias controller, not the lighter coefficient.

```python
# TokenChoiceMoE.forward — router math kept in fp32 under bf16 autocast
router_logits = self.router(flat_x.float())              # [N, E]
affinity = torch.sigmoid(router_logits)
route_scores = affinity + self.router_bias                # bias: selection only
_, top_idx = torch.topk(route_scores, K, dim=-1)
clean_scores = affinity.gather(1, top_idx)                # clean: output weights
top_weight = clean_scores / clean_scores.sum(-1, keepdim=True).clamp_min(1e-9)

# non-gradient bias controller, called once per optimizer step
@torch.no_grad()
def update_router_bias(self, load_frac):
    self.router_load_ema.mul_(beta).add_(load_frac, alpha=1 - beta)
    delta = (1/E - self.router_load_ema) * E
    self.router_bias.add_(eta * delta)
    self.router_bias.sub_(self.router_bias.mean())
    self.router_bias.clamp_(-0.25, 0.25)
```

### A.3 Exclusive self-attention (XSA)

**Idea.** Project the attention output orthogonal to the current token's own value direction. With attention output $$y_t$$ and the token's own (post-mix) value $$v_t$$:

$$y_t \leftarrow y_t - \left(y_t \cdot \hat v_t\right)\hat v_t, \qquad \hat v_t = v_t / \lVert v_t \rVert.$$

The division-of-labor argument: the residual connection already carries the token's own information perfectly, so any component of the attention output parallel to the token's own value is redundant pointwise transformation; removing it forces attention to specialize in *contextual* information orthogonal to what the token already knows. The interesting part for this project is that the win survives with a sparse FFN sitting after attention — a priori the router could have reacted badly to the modified output geometry, and it didn't.

<!-- TODO(chang-han): add the XSA paper citation -->

```python
y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
if self.exclusive_attention:
    self_v = v  # expand kv-heads to match query heads under GQA, then:
    self_v_dir = F.normalize(self_v.float(), dim=-1).to(dtype=y.dtype)
    y = y - (y * self_v_dir).sum(dim=-1, keepdim=True) * self_v_dir
```

### A.4 Headwise attention gate, near-identity init

**Idea** (from [gated attention](https://arxiv.org/abs/2505.06708)). A query-dependent sigmoid gate per head, applied after attention:

$$y^{(i)} \leftarrow \sigma\!\left(g_i(x)\right) \cdot y^{(i)}, \qquad i = 1,\dots,h,$$

with the gate weights zero-initialized and bias set so $$\sigma(\text{bias}) = 0.98$$ — the model starts at 98% of an ungated network and must *learn* to attenuate. The init bracketing was a nice micro-study: 0.95 worked, 0.98 was better, 0.99 was an exact tie with 0.98, and an exactly-centered variant ($$2\sigma(g)$$, init at identity, able to amplify) won the first 200 warmup steps and then lost — so the optimum is "almost identity, attenuation-only." A side observation I don't fully understand but reproduced across runs: the gate made *router* load cleaner, suggesting attention-side modulation relieves pressure the MoE would otherwise absorb. Per-step gains were tiny (0.05% combined); router hygiene is arguably why it stayed in the stack.

```python
self.c_attn_gate = nn.Linear(self.n_embd, self.n_head, bias=True)
# init: weight.zero_(); bias.fill_(log(0.98 / 0.02))
gate = torch.sigmoid(self.c_attn_gate(x)).view(B, T, self.n_head, 1)
y = y * gate.to(dtype=y.dtype)
```

### A.5 Dense early layers ("dense stem")

**Idea** (Codex's). The first $$L_d = 2$$ FFN layers are plain dense SwiGLU with hidden size $$K \cdot d_{\text{moe}} = 3584$$ — exactly the active width of a top-2 MoE layer, so active parameters are matched — and only layers $$\ell \geq 2$$ are sparse. Early-layer token representations are too undifferentiated for routing to be informative, so spend the early FFN budget densely and save sparsity for layers where routing has signal. Removing two 16-expert banks also cuts total parameters by 116M and visibly raises MFU. Matched-step validation (Story 3) confirmed a genuine, if small, per-step gain on top of the throughput — *at the fixed-wall horizon*. DeepSeek-V3 ships the same pattern with three dense layers; here three was tested and lost.

**Post-scaling caveat** (see "The twist"): under compute-optimal token budgets the stem is a quality *cost* at S75 and F1 (where the all-MoE ablation wins by 0.34% and 1.87% respectively) and a small net win only at F2. Its durable contributions are throughput and router stability — the all-MoE F1 run crossed the load-safety threshold in one layer, which no dense-stem run ever did. Read it as scale-dependent insurance, not as a free lunch.

```python
class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        if layer_idx < config.dense_early_layers:   # = 2
            self.ffn = DenseSwiGLU(config.n_embd, config.dense_hidden_dim)
        else:
            self.moe = TokenChoiceMoE(config)
```

## B: ideas that didn't work

For completeness, everything that was tried and rejected in the clean lineage, with the one-line reason. Learned scalar first-value residual (disturbed routing before helping; see Story 1). Learned and normalized-learned value mixes (stabilizable by the sigmoid/bias router, but quality never matched the fixed mix). Aggressive fixed mixes 2:1, 3:1 (worse matched-step loss even with a healthy router — a quality failure, not a stability one). Fine-grained experts 32/top-4/896 (lost twice, ~0.018 BPB, despite cleaner per-expert load). Sigmoid affinities *without* bias (early matched-step edge that evaporated by run end). Softmax + light load-balance control (the run that earned the router its keep). Six depth-scaling variants (every placement of $$1/\sqrt{\ell}$$ lost; internal worst, output-side least bad; consistent story: this model wants full-amplitude branches). Centered $$2\sigma$$ attention gate (won warmup, lost training). Third dense layer (saturation). Attention-gate init 0.99 (tie with 0.98, kept 0.98). E ∈ {4, 8, 32, 64, 128} (the U-curve). LRs 0.01 and 0.001 at 91M fixed-wall, 0.003 and 0.000333 at F2, 0.003 at F3, 0.0003 at F4 (the moving optimum).

Every one of these has a full pre-registered entry in the research log. The discard pile is the dataset; the leaderboard is just its argmax.

---

*The harness is adapted from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch); data is [ClimbMix](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle). Compute generously supported by VESSL AI, plus spot instances from Vast.ai. The research log and programs may end up public once I clean them up — if you'd find the raw logs useful, tell me and I'll prioritize it.*
