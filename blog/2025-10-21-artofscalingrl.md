---
layout: post
title: "The Art of Scaling Reinforcement Learning Compute for LLMs"
categories: []
year: 2025
type: paper
author: Khatri
exturl: https://arxiv.org/abs/2510.13786
---
The RL-for-LLMs landscape is, to put it mildly, highly dispersed. It's arguably the least stable training regime in open development, which is how we've ended up with a massive gap between the success of closed labs and the anemic, hard-to-replicate results in the open scene. The field has been notoriously "hill-climby," with 2025 seeing a barrage of papers focused on minor algorithmic tweaks that show empirical gains at minimal scales, making the whole space even more scattered and confusing.

We've seen parallel work on frameworks that bake in so many minor differences that it's nearly impossible to disentangle confounding variables. This makes proper ablation studies a nightmare, and it's not uncommon to see conflicting findings published back-to-back.

A lot the community sentiment touts "scaling reinforcement learning" as the lighthouse, yet provides little to no focus on *actual scaling*. The best we had before this was probably NVIDIA's ProRL paper, which at least did some prolonged training runs, but it didn't look at how different methods *scale predictively*.

This paper is the first *sane*, structured attempt to move beyond ad-hoc solutions and establish a proper methodology for scaling RL compute.

### a scaling framework
The authors' core premise is simple: to make RL training as predictable as pre-training, we need a scaling law. While pre-training loss famously follows a power law, the authors find that RL pass-rate (a bounded metric) is much more robustly modeled by a sigmoidal curve.

They fit their experiments to the following sigmoid, which breaks performance down into two key components: asymptotic performance and compute efficiency:

$$
\underbrace{R_C - R_0}_{\text{Reward Gain}} =
\underbrace{(A - R_0)}_{\text{Asymptotic Reward Gain}}
\times
\underbrace{\frac{1}{1 + (C_{\text{mid}} / C)^B}}_{\text{Compute Efficiency}}
$$

Here, **$A$** is the all-important **asymptotic pass rate**—the performance ceiling. **$B$** is the scaling exponent (compute efficiency), and **$C_{\text{mid}}$** is the compute midpoint of the curve. The methodology is to run ablations at a medium scale (e.g., 3.5k-8k GPU-hours), fit this curve to find the best $A$, combine the winning choices into a single recipe (dubbed **ScaleRL**), and then validate the curve by extrapolating it and *continuing the run* to an extreme scale (up to 100,000 GPU-hours).

### ScaleRL
The paper is a massive empirical study, but the key ablations that form ScaleRL are worth noting.

First, they tackle the asynchronous RL setup. They compare the standard **PPO-off-policy-k** (common in frameworks like `verl`) with **PipelineRL-k** (used by Magistral). In PPO-off-policy, generators and trainers run in distinct phases. In PipelineRL, generators stream traces, and trainers push updates immediately, using the new weights even on stale KV caches. As you'd expect, the asymptotic performance ($A$) is similar, but PipelineRL-k significantly boosts compute efficiency ($B$) by reducing idle time.

With PipelineRL-8 as the new baseline, they ablate several other key algorithmic choices:

* **Loss Function**: They compare the asymmetric DAPO loss against GSPO (Qwen's sequence-level GRPO) and CISPO (MiniMax's loss). Both GSPO and CISPO *substantially* outperform DAPO, posting a much higher asymptotic pass-rate $A$. They land on CISPO.
* **LM Head Precision**: MiniMax suggested using FP32 precision for the LM head to reduce numerical mismatches between inference and training kernels. The authors find this fix *dramatically* improves the asymptotic ceiling $A$. This is a key finding, especially since other teams (like Qwen) reported this *didn't* fix stability at their (much smaller) scales.
* **Loss Aggregation**: They test sample average (GRPO-style), prompt average (DAPO-style), and token average. We've had this discussion—sample average is clearly biased. The paper confirms this, finding prompt-average achieves the highest asymptotic performance.
* **Advantage Normalization**: They compare prompt-level (GRPO's group norm), batch-level, and no normalization. All three yield similar performance and scaling, so they adopt batch-level norm as it's theoretically sound.
* **Zero-Variance Filtering**: Prompts that are too easy (all pass) or too hard (all fail) have zero advantage and contribute no gradient signal. The authors confirm that filtering these prompts out of the loss calculation (as in Seed 1.5 Thinking) yields a better asymptote.
* **Data Curriculum**: They implement a simple curriculum: if a prompt's historical pass rate exceeds 90%, it's permanently dropped from subsequent epochs. This also improves both scalability and the final reward $A$.

#### stability is the *real* scaling law

Combining these into the ScaleRL recipe, the authors run Leave-One-Out (LOO) ablations. Interestingly, when combined, the individual impact of each component on the asymptote $A$ seems to shrink.

But the authors argue—and I strongly agree—that this misses the point. These components aren't just about squeezing the last drop of performance; they're about **stability** and **robustness**. RL instability is no joke, and this paper's appendix has the most telling charts.

The real story is the hyperparameter sensitivity. GRPO/DAPO-style losses are *notoriously* sensitive to the $\epsilon_{\text{max}}$ clipping ratio. The paper shows this clearly: changing $\epsilon_{\text{max}}$ doesn't just change efficiency, it *fundamentally changes the asymptotic performance $A$*. An $\epsilon_{\text{max}}$ of 0.26 gives an $A$ of 0.530, while 0.27 drops it to 0.480. This is, frankly, ridiculous. Tuning this single parameter is a scaling bottleneck in itself.

Conversely, they ablate the upper clipping ratio for CISPO and find that performance is almost identical across a wide range of values. This robustness alone is reason enough to adopt it for large-scale training. They also find GSPO is robust once you find the right scale, but it suffered from sudden training crashes, especially on larger models, which is why they ultimately preferred CISPO.

The authors also track other stability metrics. They note that training instability and crashes are almost always correlated with a rising truncation rate. Failed runs often saw truncations climb to 10-15%, while stable ScaleRL runs kept them below 5%. This gives us another critical metric to monitor.

One point of contention is entropy. The paper notes that two runs with different batch sizes had wildly different downstream AIME-24 performance but nearly identical entropy-per-step trajectories. They conclude that "simply maintaining higher entropy does not translate into better generalization". I agree with the statement, but I don't think any serious practitioner thought entropy predicted performance. We track entropy as a diagnostic for model collapse. Still, their point stands: it's not the goal, it's a health metric.

### takeaway

This is incredibly important work. It's the first structured methodology for taming RL, and it gives the open community a path to follow that isn't just "copy Magistral's config and pray." The key takeaway isn't just the final ScaleRL recipe; it's the methodology of using small-scale runs to predict large-scale asymptotes. The focus on asymptotic performance ($A$) over simple compute efficiency ($B$) is the right one for pushing the frontier.

The drawbacks are clear, and the authors admit them: this is all on one math dataset (Polaris-53K) and (mostly) one 8B model. We need to see these laws reproduced on more diverse data (e.g., code, general reasoning) and different model architectures. After the Qwen 2.5 curfuffle—where a model improved on benchmarks without a correct learning signal—we should all be wary of single-model, single-dataset results.

But even with that caveat, this paper provides the blueprint. Now, someone just has to go reproduce it.