---
layout: post
title: "Spurious Rewards: Rethinking Training Signals in RLVR"
categories: []
year: 2025
type: paper
---
The Qwen model family—particularly Qwen2.5—has become the de facto foundation for reinforcement learning research in 2025. Much of the recent hype around Reinforcement Learning from Verifiable Rewards (RLVR) has centered on Qwen2.5-Math models, with numerous studies building RLVR pipelines atop them. However, the field has grown increasingly skeptical of the actual gains conferred by RL, as recent findings suggest that models can achieve surprising improvements with just a single training sample—or even in the absence of meaningful reward signals.

Conventional wisdom in RLVR posits that high-quality supervision is crucial. But this paper challenges that assumption. It shows that Qwen2.5-Math-7B can achieve significant performance gains on MATH-500 benchmarks using "spurious" rewards—signals that are weakly informative or even misleading. In several cases, the results nearly match those obtained with ground-truth labels.

### When Nonsense Rewards Work

Starting with Qwen2.5-Math-7B, the authors demonstrate that various forms of spurious feedback can produce substantial gains:

* **Ground Truth**: Standard RLVR using correct answers leads to a 28.8% gain.
* **One-Shot RL**: Training on a single correct sample achieves a 24.4% boost.
* **Majority Vote**: Using the most common answer from 64 model samples yields a 26.5% improvement.
* **Incorrect Rewards**: Even when rewarding *wrong* answers, the model still improves by 24.6%.
* **Format Reward**: Rewarding outputs with a `\boxed{}` expression (irrespective of correctness) gives a 16.4% gain.
* **Random Reward**: Using Bernoulli(0.5) random rewards results in a 21.4% gain.

Critically, these gains are unique to Qwen models. When applied to Llama3 and OLMo2, the same reward strategies yield negligible or even negative results.

### Reward Ablation Across Models

To investigate generality, the study applies the same reward schedules to:

* Qwen2.5 base models (7B and 1.5B),
* OLMo2-7B (standard and SFT variants),
* Llama3.1-8B, Llama3.2-3B, and their Instruct variants.

The results are striking. Only Qwen models reliably benefit from spurious rewards. Other families require high-quality supervision to improve and often degrade under weak reward signals. These divergences are likely due to differences in pretraining data and reasoning priors.

### Why Does This Work?

The authors hypothesize that RLVR in Qwen models amplifies latent reasoning strategies—particularly *code reasoning*. For example:

* **Pre-RLVR**, Qwen2.5-Math-7B solves 65% of math problems by generating Python code (without code execution).
* **Post-RLVR**, regardless of reward quality, code reasoning usage rises to over 90% of completions.
* Performance increases correlate tightly with increased code usage frequency.

This is not true for other models. In Llama and Olmo, either code generation is absent or harmful. When spurious rewards encourage code usage in these models, accuracy drops.

### The Random Reward Paradox

One of the most counterintuitive findings is that even *completely random* rewards (e.g., reward = 1 if `random.random() < 0.5`) consistently improve Qwen2.5-Math performance. This led the authors to explore the internals of GRPO (Group Relative Policy Optimization) to identify why.

They isolate a key mechanism: **clipping bias** in GRPO. This bias arises because the clipping term in the loss favors responses that resemble the model's existing priors. When rewards are random, this has the unintended effect of reinforcing behaviors the model already strongly exhibits—namely, code reasoning. Ablation studies show that:

* With clipping, random rewards yield a 21% gain.
* Without clipping (via disabling or matching πₜ = πₒₗd), no reliable improvement occurs.
* Without clipping, training becomes stochastic and unstable across seeds.

Thus, the reward signal is not in the randomness itself, but in the optimization algorithm’s structure.

### Strategy Switching and Partial Attribution

The authors perform a fine-grained analysis, categorizing reasoning patterns into:

* Code → Code
* Code → Language
* Language → Code
* Language → Language

The majority of performance gain arises from *Language → Code* transitions, affirming the hypothesis that RLVR acts by reinforcing latent but effective reasoning strategies.

Interestingly, for "bad-code" models like Qwen2.5-7B and OLMo2-7B-SFT, suppressing code generation via "no Python" compound rewards *improves* performance.

### Beyond Code: The Role of Repetition

Another elicitable behavior examined is repetition. Qwen models sometimes produce repetitive outputs, but this can be countered by a simple reward: penalize responses that repeat substrings more than 10 times. This “no-repetition” reward also boosts performance in Qwen models—but not others.

### Implications

This paper makes three crucial points:

1. **Pretraining priors dominate RLVR outcomes**. Without strong base behaviors (e.g., code reasoning), even good RLVR won't help.
2. **Spurious rewards work because they reinforce these priors**, not because they teach anything new.
3. **Clipping bias acts as a signal amplifier**, allowing even random noise to become a form of curriculum for models with useful latent behaviors.

This calls into question the current RLVR benchmarking landscape. Much of it may overfit to Qwen-like models, giving a misleading picture of reward design efficacy. Future RLVR work must test across diverse base models to ensure generalizability.
