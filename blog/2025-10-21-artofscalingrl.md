---
layout: post
title: "The Art of Scaling Reinforcement Learning Compute for LLMs"
categories: []
year: 2025
type: paper
author: Khatri
exturl: https://arxiv.org/abs/2510.13786
---
[WIP]


First attempts at establishing proper scaling laws for **large scale** reinforcement learning. The study explores numerous design choices through individual runs at 8-16k GPU hours and use them to fit a sigmoid scaling curve then validate these curves through continued training.

RL is currently much more of an art than a science. There is limited work in trying to understand how different design choices **scale**. RL papers in 2025 have been extremely hill-climby, focusing on minor algorithmic tweaks that lead to empirical improvements at minimal scales making the space even more scattered. The best thing we had before this was probably the ProRL paper from NVIDIA that actually did some prolonged training runs ablating existing methods. However, no papers have looked at how different methods **scale**. The paper first ablates different design choices, fitting curves to these, then extends training to confirm these curves.

The curves use to fit the experiments are sigmoid, defined as
$$
\underbrace{R_C - R_0}_{\text{Reward Gain}} =
\underbrace{(A - R_0)}_{\text{Asymptotic Reward Gain}}
\times
\underbrace{\frac{1}{1 + (C_{\text{mid}} / C)^B}}_{\text{Compute Efficiency}}
$$
where A represents the asymptotic pass rate, B the scaling exponent that determines compute efficiency and $C_{mid}$ sets the midpoint of the RL performance curve. The following image provides an intuition for the above formula:

Note that B and C primarily affect the efficiency of the run and A denotes the asymptotic performance at large compute scale. Ablations are run at 3.5k to 4k GPU hours, if training is stable it is run for longer. The best ablation choices are combined into a **ScaleRL** run. 

**personal thoughts**. While helpful, these are far from the deeply insightful power law relating downstream test loss to pretraining compute. The expected reward we see here is a bounded metric. The RL scaling laws are most useful for ablating design choices.  

#### async RL

The paper first considers two choices of async off-policy RL setup: PPO-off-policy-k and PipelineRL-k.

**PPO-off-policy-k**  is the default approach of asynchronous RL and the standard approach in the commonly used verl framework. In this setup the old policy generates reasoning traces for a batch of B prompts. Each gradient update process a mini-batch of B' prompts, resulting in k = B/B' gradient updates per batch. 

 **PipelineRL-k** is a recent approach suggested in things like Magistral and AReaL. In this regime, generators continously produce reasoning traces in a streaming fashion. Whenever trainers finish a policy update, the new parameters are immediately pushed to the generators, which continue generating with the updated weights but a stale KV cache from the old policy. Once a full batch of traces are generates it is passed to the trainers for the next update. The paper introduces a new parameter k: the trainers wait if they get k steps ahead of the generators. 

As expected, the asymptotic performance A is fairly similar across different levels of off-policy but increasing k improves compute efficiency B; with PipelineRL-8 improving efficiency further. 

#### algorithmic choices

the study continues by ablating several algorithmic choices. 

Starting with loss type they compare the asymmetric DAPO loss with GSPO (qwen sequence level GRPO) and CISPO (miniMax). Both GSPO and CISPO substantially otperform DAPO, improving the asympotic pass rate A by a large margin. The authors adopt CISPO as the best loss type.

MiniMax suggests adopting FP32 precision for the LM head to reduce numerical mismatches between inference and training engine token probabilities. The authors find that this fix drastically improves A. This is somewhat surprising, the Qwen team found that this precision fix did not fix training stability, but their experiments were at a much lower scale.

For loss aggregation they compare sample average where each rollout contributes equally (GRPO standard), prompt average where each prompt contributes equally (as in DAPO) and token average where all token losses in a batch are averaged directly without intermediate grouping. Both prompt average and token average perform similarly asymptotically, they adopt prompt average. We've had this discussion many times and we repeatedly find that the standard sample average is too biased against the sample length, difficulty etc.

Advantage normalization is something that has also been discussed in a few papers, this can be performed at the prompt level where advantages are normalized by the standard deviation of rewards from rollouts of the same prompt (that is group normalization as in GRPO), batch level where advantages are normalized by the std across generatiosn in a batch, or no normalization (as proposed in Dr.GRPO). They find that these yield similar performance at scale, with similar scaling characteristics and final A. They adopt batch-level norm as it is theoretically sound.

Prompts that yield identical rewards, typically prompts that are on the difficulty extremes, that is either too easy or too hard yield no learning signal under GRPO as they have zero advantage. GRPO includes such prompts in loss computation. The authors ablate what happens if these prompts are dropped completely in the loss calculation, as proposed in Seed 1.5 Thinking. They find that this yields a drastic asymptotic improvement.

Finally, they adopt a data curriculum strategy by storing a history of pass rates for all prompts, when the prompt pass rate are above 0.9 it is dropped from the data in subsequent epochs. This also improves scalability and the asymptotic reward A.

#### scaleRL

the above methods are combined to form a single recipe, termed scalerl. Leave-one-out (LOO) ablations are performed on this recipe: starting from ScaleRl, they revert one axis at a time to the baseline (GRPO / asymmetric DAPO). This ensures that each design choice contributes positively even in the presence of all others. While the individual design choices had clear impact on asymptotic choices in the ablations, in the LOO experiments the components appear less ciritical individually as most LOO ablations result in similar asymptotic rewards. However, the authors argue that even if the choice seems redundant in the combined recipe, all components provide stability and robustness that are not directly evident in the scaling curves. In summary, even when individual design choices appear redundant within the combined recipe, they often enhance training stability, robustness, or efficiency in ways that generalize across models and setups. 

The ScaleRL recipe provides clean, predictable scaling behavior across a bunch of different axis. The authors look at scaling knows such as generation length, batch size and model scale, finding that across all these, they are able to fit the saturating power law equation early in training (precisely at half target budget) extrapolate these to the target budget and extend training and verify the forecast.

This is taken to extreme with a single ScaleRL training run lasting 100k hours, with predicitive scaling behaviour. 

#### key findings

**Compute scaling extrapolation**. Using smaller-scale ablations in a systematic way to predict performance at larger scales enables more effective experimentation in finding the final scalable recipe.

**Key algorithmic choices**. The off-policy algorithm, loss function (CISPO) and model precision (FP32 LM head) are the most importance decisions in this papers ablations. 

**Asymptotic performance vs efficiency**. Many ablations in this paper found options that improve both efficiency and asymptotic performance, but this is not always the case. This paper opts for asymptotic performance first and foremost, with the goal of finding choices that scale best. 

#### appendix

**generation length**
controlling generation length is important for training efficiency and stability. the paper considers two approaches

1. interruptions, used in GLM 4.1V and Qwen3 where model thinking is forcibly stopped by appending a marker phrase such as "Okay, time is up. Let me stop thinking and formulate a final answer </think>", signaling the model to terminate its reasoning and produce a final answer. 
2. Length penalties as introduced in DAPO where overly long completions are penalized. This penalty is only added to correct traces, discouraging excessively long generations. 

comparing the two they find that interruptions lead to slightly better asymptotic reward.

**entropy**
a lot of work surrounding RL has discussed the importance of retaining entropy as it is a direct measure of the models exploration capacity. Specifically entropy collapse has been observed to be a key determinant of training instability / divergence. In this work, the authors find that two different training runs with clearly different downstream performance (AIME 24) show almost exactly the same entropy trajectories. The authors point out that "although entropy is sometimes used as a proxy for exploration, simply
maintaining higher entropy does not translate into better generalization". I agree with this point, entropy is not a predictor of downstream performance, but I don't think any practitioner thought this way either. Rather entropy is a very good metric to look at as a way to predict model collapse. Also, given the lack of generalization tests in this paper I wouldn't draw too many conclusions on their entropy findings. 

**truncations**
across experiments the authors found that training instabilities were often linked to truncations. AS generation length grew, many RL runs  exhibited fluctuating truncation rates that sometimes increased over training. Many failed runs also saw truncation rates that grew into the 10-15% range. In contrast, ScaleRl runs that were stable had truncations remaining below 5% for 90% of training. This provides insight to yet another key metric to track during RL training that could be determinent of training instability.

**loss type - stability and robustness**

GRPO/DAPO-style losses are highly sensitive to the choice of clippping ratio hyperparameter $\epsilon_{\text{max}}$. This goes in line with what we've seen in previous papers, with the upper clipping parameter having a massive impact on performance. Ablating this

we see that $\epsilon_{\text{max}}$ has an impact far beyond many other hyperparameters, having a direct impact on the terminal reward A. This is a striking effect, and one that is very undesirable giving the extreme sensitivity of this single parameter on this single dataset pass rate. 

Similar ablations are run for the higher clipping ratio for CISPO, keeping the lower clipping ratio fixed at 0. 


Across a wide range of values there is little difference in performance, indicating the CISPO is much more robust to this parameter. This is great, and very desirable. This alone makes a strong argument for moving away from the above GRPO/DAPO style loss formulation. This makes CISPO a strong candidate for default use in large-scale training.

GSPO is fairly robust to clipping ratio once you've identified the correct scale of the clipping ratio. 

However, the authors also note that they encounter stability issues with GSPO. GSPO would suddenly crash mid-training leading to sudden drops in performance. Restarting from a stable checkpoint allowed recovery, at least for 8B models, but this strategy failed for larger models. They were unable to determine the reason behind this. This seems to be the main reason why the adopted CISPO as their loss type in the ScaleRL recipe.
#### personal thoughts

The RL landscape is highly disperse, it's probably the least stable training regime in terms of open developement. The effect of this is that the RL success of closed labs has been nutoriously difficult to replicate in the open scene. There is a lot of parallel work on frameworks / libraries for RL training that have developed in parallel instead of in unison, baking in many minor algorithmic differences over time that make it hard to disentangle and understand. These baked in differences make proper ablation studies difficult to compare from one study to another, it is not uncommon to see quite conflicting findings on published work. 

A lot of RL work has been minor algorithmic tweaks, with fairly low scale regime empirical evidence to back it up, with quite little focus on the scale of things, all while we simultaneously tout **scaling reinforcement learning** as the only lighthouse we have to go by. There has been some work in trying to disentangle all the algorithmic nuances that have been introduced (mainly this year) and build an understanding of what actually is necessary for prolonged RL training, thinking of the ProRL paper mostly. But even that work doesn't focus on how these methods scale with compute, and if they scale predicatively. For exactly this reason, this work is important.

RL instability is no joke, it is **very** hard to train RL at larger scales.

Overall, I think this is very important work. It's the first sane, or perhaps I should say structured, attempt at understanding which of these methods scale. Just look at the ridiculous sensitivity of the DAPO upper clipping parameter. Attempting to formalize how these methods scale is very important for the open community, because the open community lags far behind the closed labs in ability to scale RL. The taken approach is also very mature, providing strong evidence for the findings, and spending a lot of compute to verify the proposed scaling curves. This is summarized well in the paper by the authors themselves: 
	
	While RL compute for LLMs has scaled massively, our understanding of how to scale RL has not kept pace;
	the methodology remains more art than science. Recent breakthroughs in RL are largely driven by isolated
	studies on novel algorithms (e.g., Yu et al. (DAPO, 2025)) and model-specific training reports, such as,
	MiniMax et al. (2025) and Magistral (Rastogi et al., 2025). Critically, these studies provide ad-hoc solutions
	tailored to specific contexts, but not how to develop RL methods that scale with compute. This lack of scaling
	methodology stifles research progress: with no reliable way to identify promising RL candidates a priori,
	progress is tied to large-scale experimentation that sidelines most of the academic community.

Drawbacks of the paper are clear, and clearly realized by the authors themselves, all experiments are run on a single math dataset which leads to questions about the generealizability of these laws. We need to see these laws reproduced on a wider variety of data. The same can be said for the model. We've already had a big RL curfuffle this year with base models (Qwen 2.5 improving on benchmarks without a correct learning signal), all of these RL experiments should be performed across atleast 3 base models just to confirm that the findings generalize to three different pretrained models and that they are not the effect of something in the pretraining data.  But, they've already spent $4.2 million on experiments in the paper so I can imagine research leads were already pushing the budget as is.