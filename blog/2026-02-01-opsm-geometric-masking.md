---
layout: post
title: "deepseek off policy sequence masking is a geometric sequence mask"
categories: [RL]
year: 2025
type: blog
---

DeepSeek's Off-Policy Sequence Masking (OPSM), introduced in the V3.2 technical report, is mathematically equivalent to a one-sided geometric sequence mask. If you're implementing OPSM, this means you can reuse existing geometric masking machinery—the "KL threshold" in OPSM is just the log of a geometric mean. This post walks through the theory that leads to this equivalence.

## background: the ppo ratio

PPO's clipped surrogate objective contains a ratio between policies:

$$
J(\theta) = \min\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}A, \text{clip} \left( \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}, 1-\varepsilon, 1+\varepsilon \right) A \right)
$$

This ratio emerges from importance sampling. The policy gradient requires expectations over trajectories from the current policy $\pi_\theta$, but we want to take multiple gradient steps on a batch collected from a fixed policy $\pi_{\theta_{\text{old}}}$. The importance weight $\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}$ corrects for this distribution mismatch.

This correction enables multiple gradient steps per batch, which is crucial for efficiency. Generating rollouts is expensive—we want to squeeze as much learning as possible from each generation pass.

## the three policy formulation

Standard PPO conflates two distinct roles into a single "old" policy. The batch-size invariant PPO paper (Schulman et al.) decouples these into separate policies:

**Behavior policy** ($\mu$): The policy that actually generated the data. It appears in the importance sampling ratio that corrects for off-policy data:

$$
\frac{\pi_{\theta}(a|s)}{\mu(a|s)}
$$

**Proximal policy** ($\pi_{\text{old}}$): A recent policy used as an anchor for the trust region. It appears in the clipping term that prevents the policy from changing too drastically:

$$
\text{clip}\left(\frac{\pi_{\theta}(a|s)}{\pi_{\text{old}}(a|s)}, 1-\epsilon, 1+\epsilon\right)
$$

In standard PPO, $\mu = \pi_{\text{old}}$—the same policy serves both roles. The decoupled formulation separates them:

$$
J(\theta)_{\text{decoupled}} = \frac{\pi_{\text{old}}(a | s)}{\mu(a | s)} \min\left(\frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}A, \text{clip} \left( \frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}, 1-\varepsilon, 1+\varepsilon \right) A \right)
$$

The leading term $\frac{\pi_{\text{old}}}{\mu}$ handles off-policy correction. The ratio inside the min/clip handles trust region enforcement. These are separate concerns.

GRPO inherits this structure. It replaces the learned value function with Monte Carlo advantage estimates from grouped completions, but retains the clipped surrogate objective.

## why this matters for llm training

Modern RL training for LLMs uses disaggregated systems: a high-throughput inference engine (vLLM, SGLang) generates rollouts, while a separate training framework performs gradient updates. This separation exists because inference engines are optimized for generation throughput, training frameworks for gradient computation.

This introduces two sources of off-policyness:

**Training-inference mismatch**: The sampling distribution from the inference engine ($\mu$) may differ from the training model ($\pi_{\text{old}}$) due to quantization, different attention implementations, numerical precision, or softmax implementations. Even with identical weights, the distributions can diverge.

**Policy staleness**: RL systems generate large rollout batches, then split them into mini-batches for multiple gradient steps. By the time you're training on the last mini-batch, the policy has changed. The data is stale relative to the current $\pi$.

The three policy formulation gives us language to discuss these separately. We track three sets of log-probs:

| Symbol | Variable | Description |
|--------|----------|-------------|
| $\pi$ | `per_token_logps` | Current policy being optimized |
| $\pi_{\text{old}}$ | `old_per_token_logps` | Training model at rollout time (proximal policy) |
| $\mu$ | `sampling_per_token_logps` | Inference engine at rollout time (behavior policy) |

The ratio $\frac{\pi_{\text{old}}}{\mu}$ captures training-inference mismatch. The ratio $\frac{\pi}{\pi_{\text{old}}}$ captures policy staleness. Standard PPO clipping handles the latter. The former requires explicit importance sampling correction.

## importance sampling correction

Given samples from behavior policy $\mu$, we estimate expectations under target policy $\pi$ using importance weights:

$$
\mathbb{E}_{y \sim \pi}[f(y)] = \mathbb{E}_{y \sim \mu}\left[\frac{\pi(y)}{\mu(y)} f(y)\right]
$$

For autoregressive generation, the sequence-level importance ratio is a product of per-token ratios:

$$
\frac{\pi(y|x)}{\mu(y|x)} = \prod_{t=0}^{T-1} \frac{\pi(y_t | x, y_{<t})}{\mu(y_t | x, y_{<t})}
$$

Define the per-token ratio as $\rho_t = \frac{\pi(y_t | x, y_{<t})}{\mu(y_t | x, y_{<t})}$, making the sequence-level ratio $\rho_{\text{seq}} = \prod_t \rho_t$.

To address training-inference mismatch specifically, we apply IS correction between $\pi_{\text{old}}$ and $\mu$:

$$
\rho_t^{\text{TI}} = \frac{\pi_{\text{old}}(y_t | x, y_{<t})}{\mu(y_t | x, y_{<t})}
$$

In practice, we bound these ratios to maintain stability. Two approaches:

**Truncated Importance Sampling (TIS)** clips ratios:
$$
\rho \leftarrow \text{clip}(\rho, C_{\min}, C_{\max})
$$

**Masked Importance Sampling (MIS)** discards samples with extreme ratios:
$$
\rho \leftarrow \begin{cases} \rho & \text{if } C_{\min} \leq \rho \leq C_{\max} \\ 0 & \text{otherwise} \end{cases}
$$

Both can be applied per-token or per-sequence. The choice matters.

## the length bias problem

The sequence-level importance ratio $\rho_{\text{seq}} = \prod_t \rho_t$ grows or shrinks exponentially with sequence length.

Consider a modest per-token drift of 0.1%—$\rho_t \approx 1.001$ on average. Over 2000 tokens:

$$
\rho_{\text{seq}} = (1.001)^{2000} \approx 7.39
$$

Even with minimal per-token divergence, long sequences get systematically rejected or heavily down-weighted. The effective trust region varies dramatically with sequence length, and the model may learn to generate shorter sequences to avoid penalties.

## geometric sequence masking

Geometric Sequence Masking uses the geometric mean of importance ratios instead of their product:

$$
\rho_{\text{geo}} = \left( \prod_{t=0}^{T-1} \rho_t \right)^{1/T}
$$

In log-space, this is the arithmetic mean of log-ratios:

$$
\log \rho_{\text{geo}} = \frac{1}{T} \sum_{t=0}^{T-1} \log \rho_t = \frac{1}{T} \sum_{t=0}^{T-1} \log \frac{\pi(y_t|x, y_{<t})}{\mu(y_t|x, y_{<t})}
$$

This quantity represents the average per-token log-likelihood ratio. The key property is length invariance: the geometric mean doesn't scale with $T$.

The geometric sequence mask applies a threshold:

$$
g_{\text{geo-mask}} = \mathbf{1}\left[ C_{\min} \leq \rho_{\text{geo}} \leq C_{\max} \right]
$$

In code, starting from the standard IS calculation for training-inference mismatch:

```python
# Variable mapping:
#   old_per_token_logps = π_old (training model at rollout)
#   sampling_per_token_logps = μ (inference engine)
#   mask = attention mask for valid tokens

# Per-token log importance ratio: log(π_old / μ)
per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * mask

# Sequence-level: sum of log-ratios = log of product
per_seq_logps_diff = per_token_logps_diff.sum(dim=-1, keepdim=True)

# Geometric mean: divide by sequence length
# This is the only change from standard sequence-level IS
seq_lengths = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
geo_mean_log_ratio = per_seq_logps_diff / seq_lengths

# Convert back to ratio space
geo_importance_ratio = torch.exp(geo_mean_log_ratio)
```

That division by sequence length transforms length-biased sequence IS into length-invariant geometric masking.

## deepseek opsm

DeepSeek-V3.2 introduces Off-Policy Sequence Masking to stabilize training. The masking rule (using DeepSeek's 1-indexed notation):

$$
M_{i,t} = \begin{cases}
0 & \text{if } \hat{A}_{i,t} < 0 \quad \text{and} \quad \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \log \frac{\pi_{\text{old}}(o_t | q, o_{<t})}{\pi_\theta(o_t | q, o_{<t})} > \delta \\
1 & \text{otherwise}
\end{cases}
$$

A sequence is kept if either: (1) it has non-negative advantage, or (2) its average KL divergence is within threshold $\delta$.

From the paper:

> Models benefit the most by learning from its own mistakes, whereas highly off-policy negative samples can be detrimental, potentially misleading or destabilizing the optimization process.

This is one-sided: it only masks negative-advantage samples that have drifted too far. OPSM operates at sequence-level and complements GRPO's token-level ratio clipping.

A crucial detail from DeepSeek:

> Note that $\pi_{\text{old}}$ here denotes the sampling probability directly returned by the inference framework, thus the KL divergence between the old and current policy accounts for both sources of off-policyness mentioned above.

DeepSeek uses the inference engine's log-probs directly. In our notation, their $\pi_{\text{old}}$ is our $\mu$. OPSM addresses both training-inference mismatch and policy staleness in a single ratio.

## opsm is geometric sequence masking

The OPSM KL term is exactly the negative of the log geometric mean:

$$
\underbrace{\frac{1}{T} \sum_{t=0}^{T-1} \log \frac{\mu(y_t | x, y_{<t})}{\pi(y_t | x, y_{<t})}}_{\text{OPSM KL term}} = -\underbrace{\frac{1}{T} \sum_{t=0}^{T-1} \log \frac{\pi(y_t | x, y_{<t})}{\mu(y_t | x, y_{<t})}}_{\log \rho_{\text{geo}}}
$$

The condition $\frac{1}{T} \sum_t \log \frac{\mu}{\pi} > \delta$ is equivalent to:

$$
-\log \rho_{\text{geo}} > \delta \quad \Leftrightarrow \quad \rho_{\text{geo}} < e^{-\delta}
$$

OPSM is a one-sided geometric sequence mask that drops sequences where the current policy assigns significantly lower probability than the sampling distribution.

| Aspect | Geometric Sequence Mask | DeepSeek OPSM |
|--------|------------------------|---------------|
| Direction | Two-sided ($C_{\min} \leq \rho_{\text{geo}} \leq C_{\max}$) | One-sided ($\rho_{\text{geo}} \geq e^{-\delta}$) |
| Advantage condition | None | Only applies when $\hat{A} < 0$ |
| Ratio | $\frac{\pi_{\text{old}}}{\mu}$ (training-inference only) | $\frac{\pi}{\mu}$ (both sources) |

OPSM conflates both sources of off-policyness into a single ratio. We can factor this to understand what it captures:

$$
\frac{\pi(y_t | x, y_{<t})}{\mu(y_t | x, y_{<t})} = \underbrace{\frac{\pi_{\text{old}}(y_t | x, y_{<t})}{\mu(y_t | x, y_{<t})}}_{\text{training-inference}} \times \underbrace{\frac{\pi(y_t | x, y_{<t})}{\pi_{\text{old}}(y_t | x, y_{<t})}}_{\text{staleness}}
$$

This factorization is a conceptual decomposition, not how DeepSeek implements it—they use the inference engine's log-probs directly. But it clarifies that OPSM's single ratio captures both phenomena.

Taking logs and averaging:

$$
\log \rho_{\text{geo}}^{\text{OPSM}} = \underbrace{\frac{1}{T} \sum_{t=0}^{T-1} \log \frac{\pi_{\text{old}}}{\mu}}_{\text{training-inference term}} + \underbrace{\frac{1}{T} \sum_{t=0}^{T-1} \log \frac{\pi}{\pi_{\text{old}}}}_{\text{staleness term}}
$$

The training-inference term is constant per rollout. The staleness term changes with each gradient step as $\pi$ updates. If you want to track these separately for debugging or ablations:

```python
# Variable mapping:
#   per_token_logps = π (current policy)
#   old_per_token_logps = π_old (training model at rollout)
#   sampling_per_token_logps = μ (inference engine)

# --- Computed once per rollout ---
# Training-inference mismatch: log(π_old / μ)
ti_log_ratio = (old_per_token_logps - sampling_per_token_logps) * mask
ti_mean = ti_log_ratio.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)

# --- Computed per gradient step ---
# Policy staleness: log(π / π_old)
staleness_log_ratio = (per_token_logps - old_per_token_logps) * mask
staleness_mean = staleness_log_ratio.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)

# Total: log(π / μ) = log(π_old / μ) + log(π / π_old)
total_geo_log_ratio = ti_mean + staleness_mean

# OPSM condition: mask if negative advantage AND high KL
# KL term in OPSM = -total_geo_log_ratio (note the sign flip)
opsm_kl = -total_geo_log_ratio
is_neg_adv = advantages < 0
is_high_kl = opsm_kl > delta
opsm_mask = ~(is_neg_adv & is_high_kl)
```

## summary

The mathematical relationships:

1. **Sequence IS** uses $\prod_t \rho_t$—length-biased
2. **Geometric masking** uses $(\prod_t \rho_t)^{1/T}$—length-invariant
3. **OPSM KL term** equals $-\log \rho_{\text{geo}}$
4. **OPSM** = one-sided geometric mask on $\frac{\pi}{\mu}$, conditioned on negative advantages
5. **Factored**: $\frac{\pi}{\mu} = \frac{\pi_{\text{old}}}{\mu} \times \frac{\pi}{\pi_{\text{old}}}$

The insight that OPSM is a geometric sequence mask connects it to a principled framework for length-invariant importance sampling and clarifies what the method optimizes.
