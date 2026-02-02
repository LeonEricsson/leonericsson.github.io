---
layout: post
title: "a geometric view of<br>off-policy sequence masking"
year: 2026
type: blog
---

Distribution shift between training and inference frameworks was a recurring theme towards in late 2025. For whatever reason, I've found myself quite interested in this problem. Every time a new model drops, I make a point of noting how the authors handle it. The issue is deceptively simple and surprisingly stubborn. It also comes at a time when scaling RL infrastructure has been a top priority for labs, with most publishing their own asynchronous training frameworks. Training stability has always been difficult in RL, and understanding it in asynchronous settings is harder still. Training-inference mismatch only makes diagnosis harder. Perhaps you've read the the Thinking Machines blog on [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) (if not, it's good, you should!). I suspect that work was partly inspired by encounters with this very mismatch. The blog addresses nondeterminism in inference engines more broadly, but it captures what's happening in RL as well: modern inference engines and training frameworks have diverged enough that they no longer produce the same outputs, even under identical weights.

Once you accept this reality, the options narrow quickly. You can try to eliminate the discrepancy at the source: roll your own inference framework, control every kernel and numerical detail, and accept a large engineering burden in exchange for consistency. Or you can accept that some mismatch is inevitable and correct for it algorithmically. The latter has been far more common in open research, and is the focus of this post.

I'll walk through the standard tools for handling off-policy data, show why naive approaches break down for long sequences, and arrive at geometric sequence masking as a length-invariant alternative. This framing also turns out to be useful for understanding DeepSeek’s Off-Policy Sequence Masking. While OPSM will appear to take a different approach, it fits naturally into the same geometric perspective developed later in the post.

This post owes a lot to Yingru Li's rigorous work on importance-sampling gradient estimators; references at the end.

## background

PPO's clipped surrogate objective contains a ratio between policies:

$$
J(\theta) = \min\left(\frac{\pi(a|s)}{\pi_{\text{old}}(a|s)}A, \text{clip} \left( \frac{\pi(a|s)}{\pi_{\text{old}}(a|s)}, 1-\varepsilon, 1+\varepsilon \right) A \right)
$$

(Throughout this post, I'll write $\pi$ rather than $\pi_\theta$ to denote the parameterized policy)

This ratio emerges from importance sampling. The policy gradient is written as an expectation under the *current* policy $\pi$, but in practice we want to take multiple optimizer steps on trajectories that were generated earlier, under some fixed rollout policy $\pi_{\text{old}}$. The importance weight $\frac{\pi}{\pi_{\text{old}}}$ corrects for that mismatch.

This correction enables multiple gradient steps per batch. Rollouts are expensive; we want to extract as much learning as possible from each generation pass, ammortizing the sampling cost aggressively.

#### the three policy formulation

Standard PPO conflates two distinct roles into a single "old" policy. The batch-size invariant PPO paper ([Schulman et al. 2021](https://arxiv.org/abs/2110.00641)) pulls these apart into separate policies:

**Behavior policy** ($\mu$): The policy that actually generated the data. It appears in the importance sampling ratio that corrects for off-policy data:

$$
\frac{\pi(a|s)}{\mu(a|s)}
$$

**Proximal policy** ($\pi_{\text{old}}$): A recent policy used as an anchor for the trust region. It appears in the clipping term that prevents the policy from changing too drastically:

$$
\text{clip}\left(\frac{\pi(a|s)}{\pi_{\text{old}}(a|s)}, 1-\epsilon, 1+\epsilon\right)
$$

In classic PPO, $\mu = \pi_{\text{old}}$—the same policy plays both roles. The decoupled formulation separates them:

$$
J(\theta)_{\text{decoupled}} = \frac{\pi_{\text{old}}(a | s)}{\mu(a | s)} \min\left(\frac{\pi(a|s)}{\pi_{\text{old}}(a|s)}A, \text{clip} \left( \frac{\pi(a|s)}{\pi_{\text{old}}(a|s)}, 1-\varepsilon, 1+\varepsilon \right) A \right)
$$

The leading term $\frac{\pi_{\text{old}}}{\mu}$ handles off-policy correction. The ratio inside the min/clip handles trust region enforcement. They address different failure modes, and as we'll see, this separation is imporant in recent RL.

GRPO adopts vanilla PPO's clipped surrogate objective without this explicit separation—assuming $\mu = \pi_{\text{old}}$. I first encountered the three-policy formulation for GRPO in AReaL,which used it in an async RL setup where multiple policy versions generate training data. But it really came into focus after [Fengyao's blog post](https://fengyao.notion.site/off-policy-rl), which applied it specifically to training-inference mismatch in disaggregated systems.

## training-inference mismatch

Modern RL training for LLMs uses disaggregated systems: a high-throughput inference engine (vLLM, SGLang) generates rollouts, while a separate training framework performs gradient updates. This split is rational—one stack is optimized for generation throughput, the other for backward-pass throughput—but this disconnect has led the two stacks to diverge in numerics and implementation details enough that they no longer induce the same output distribution under the same policy (weights).

Th [MiniMax M1 paper (sec 3.2)](https://arxiv.org/abs/2506.13585) noted this discrepancy, but it wasn't until [Fengyao's blog post](https://fengyao.notion.site/off-policy-rl) that the problem was formalized, studied systematically, and paired with concrete correction strategies. The [Ant Group paper](https://arxiv.org/abs/2510.18855) expanded on this with additional masking strategies. The common thread over the later half of 2025 has been that in disaggregated RL, we're not just dealing with policy staleness from multiple gradient steps—we're also dealing with a fundamental mismatch between the sampling distribution and the training distribution.

This gives us two sources of off-policyness:

**Training-inference mismatch**: The sampling distribution from the inference engine ($\mu$) differs from what the training model ($\pi_{\text{old}}$) would have produced due numerical precision, kernel implementation, or kernel non-determinism. 

**Policy staleness**: RL systems generate large rollout batches, then split them into mini-batches for multiple gradient steps. By the time you're training on the last mini-batch, the policy $\pi$ has moved from $\pi_{\text{old}}$.

The three policy formulation gives us language to discuss these separately. We track three sets of log-probs:

| Symbol | Variable | Description |
|--------|----------|-------------|
| $\pi$ | `per_token_logps` | Current policy being optimized |
| $\pi_{\text{old}}$ | `old_per_token_logps` | Training model at rollout time (proximal policy) |
| $\mu$ | `sampling_per_token_logps` | Inference engine at rollout time (behavior policy) |

The ratio $\frac{\pi_{\text{old}}}{\mu}$ captures the general distribution shift between the proximal policy and the rollout policy, specifically the training-inference mismatch we'll focus on here. The ratio $\frac{\pi}{\pi_{\text{old}}}$ captures policy staleness, which is what PPO's clipping mechanism constrains.

## importance sampling correction

So how do we actually correct for training-inference mismatch? The standard tool is importance sampling. If we have samples from a behavior policy $\mu$ but want to estimate expectations under a target policy $\pi$, we reweight each sample by the ratio of how likely it is under the target:

$$
\mathbb{E}_{y \sim \pi}[f(y)] = \mathbb{E}_{y \sim \mu}\left[\frac{\pi(y)}{\mu(y)} f(y)\right]
$$

For autoregressive generation, computing this ratio involves a product over tokens. Each token’s probability is conditioned on the prefix, so the sequence-level ratio factorizes as:

$$
\frac{\pi(y|x)}{\mu(y|x)} = \prod_{t=0}^{T-1} \frac{\pi(y_t | x, y_{<t})}{\mu(y_t | x, y_{<t})}
$$

We can write this more compactly by defining $\rho_t = \frac{\pi(y_t | x, y_{<t})}{\mu(y_t | x, y_{<t})}$ for the per-token ratio, giving us $\rho_{\text{seq}} = \prod_t \rho_t$ for the sequence.

For training-inference mismatch specifically, we want to correct between the training model $\pi_{\text{old}}$ and the inference engine $\mu$:

$$
\rho_t^{\text{TI}} = \frac{\pi_{\text{old}}(y_t | x, y_{<t})}{\mu(y_t | x, y_{<t})}
$$

In practice, raw importance weights are a high-variance estimator. A single trajectory with an extreme ratio can dominate the gradient and destabilize training. Two prominent approaches bound these ratios, corresponding to soft and hard trust regions:

**Truncated Importance Sampling (TIS)** clips ratios to a range, keeping the gradient signal but limiting its magnitude:

$$
\rho \leftarrow \text{clip}(\rho, C_{\min}, C_{\max})
$$

**Masked Importance Sampling (MIS)** discards samples with extreme ratios entirely, enforcing a hard trust region:

$$
\rho \leftarrow \begin{cases} \rho & \text{if } C_{\min} \leq \rho \leq C_{\max} \\ 0 & \text{otherwise} \end{cases}
$$

As usual with RL, the choice between the two *depends*. Clipping treats every sample as fundamentally usable, and interprets extreme ratios as a variance problem. But extreme ratios typically arise when $\mu(y_t)$ is near the numerical precision floor, producing out-of-distribution samples where importance sampling breaks down. Clipping such a sample to $C_{\max}$ still includes it in the gradient update, which can introduce persistent bias rather than just variance.

Masking takes the opposite stance. It defines an explicit trust region and discards samples that violate it. The tradeoff is sample efficiency versus robustness: TIS tends to work well when mismatch is mild and the sampler is well-calibrated, while MIS is safer when out-of-distribution behavior is common. As we will see, long sequences make those tail events much more likely.

Both can be applied per-token (checking each $\rho_t$) or per-sequence (checking the product $\rho_{\text{seq}}$). In code, a sequence-level MIS check looks like:

```python
# Variable mapping:
#   old_per_token_logps = π_old (training model)
#   sampling_per_token_logps = μ (inference engine)

# Per-token log ratio: log(π_old / μ)
per_token_log_ratio = (old_per_token_logps - sampling_per_token_logps) * mask

# Sequence-level: sum of logs = log of product
seq_log_ratio = per_token_log_ratio.sum(dim=-1)
seq_ratio = torch.exp(seq_log_ratio)

# MIS: mask sequences outside bounds
mis_mask = (seq_ratio >= C_min) & (seq_ratio <= C_max)
```

Sequence-level is the [theoretically sound approach](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda) but it has a problem.

## the length bias problem

The issue is fundamental: the importance ratio $\rho_{\text{seq}} = \prod_t \rho_t$ grows or shrinks exponentially with sequence length.

Consider a modest per-token drift of 0.1%—$\rho_t \approx 1.001$ on average. Over 2000 tokens:

$$
\rho_{\text{seq}} = (1.001)^{2000} \approx 7.39
$$

Even with minimal per-token divergence, long sequences get systematically rejected or heavily down-weighted. The effective trust region becomes sequence-length dependent: a bound like $C_{\max} = 2$ might accept most 100-token sequences but reject almost all 2000-token sequences. This creates a bias toward shorter generations regardless of their quality.

## geometric sequence masking

The fix is to use a length-invariant measure of divergence. Instead of the product of per-token ratios, Geometric Sequence Masking uses their geometric mean:

$$
\rho_{\text{geo}} = \left( \prod_{t=0}^{T-1} \rho_t \right)^{1/T}
$$

In log-space, the geometric mean becomes an arithmetic mean, which is easier to compute. For training-inference mismatch specifically, this gives us:

$$
\log \rho_{\text{geo}}^{\text{TI}} = \frac{1}{T} \sum_{t=0}^{T-1} \log \frac{\pi_{\text{old}}(y_t|x, y_{<t})}{\mu(y_t|x, y_{<t})}
$$

This is just the average per-token log-likelihood ratio between the training model and the inference engine. Crucially, it doesn't scale with $T$—a 100-token sequence and a 2000-token sequence with the same average per-token divergence will have the same $\rho_{\text{geo}}^{\text{TI}}$.

The geometric sequence mask then applies bounds on this length-invariant ratio:

$$
g_{\text{geo-mask}}^{\text{TI}} = \mathbf{1}\left[ C_{\min} \leq \rho_{\text{geo}}^{\text{TI}} \leq C_{\max} \right]
$$

In code, we can extend the MIS calculation from before. The only change is dividing by sequence length before checking bounds:

```python
# Same setup as MIS: log(π_old / μ)
per_token_log_ratio = (old_per_token_logps - sampling_per_token_logps) * mask
seq_log_ratio = per_token_log_ratio.sum(dim=-1)

# Geometric mean: divide by sequence length to get log ρ_geo^TI
seq_lengths = mask.sum(dim=-1).clamp(min=1.0)
geo_mean_log_ratio = seq_log_ratio / seq_lengths  # <-- key change
geo_ratio = torch.exp(geo_mean_log_ratio)  # ρ_geo^TI

# Same masking logic, now length-invariant
geo_mask = (geo_ratio >= C_min) & (geo_ratio <= C_max)
```

## deepseek opsm

[DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) introduces Off-Policy Sequence Masking. Like MIS, it's a sequence-level mask that discards samples based on a divergence threshold. But OPSM has two key differences: it measures divergence between the current policy $\pi$ and the sampling distribution (not between $\pi_{\text{old}}$ and $\mu$), and it only applies to negative-advantage samples.

The masking rule:

$$
M_{i,t} = \begin{cases}
0 & \text{if } \hat{A}_{i,t} < 0 \quad \text{and} \quad \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \log \frac{\pi_{\text{old}}(o_t | q, o_{<t})}{\pi(o_t | q, o_{<t})} > \delta \\
1 & \text{otherwise}
\end{cases}
$$

A sequence is kept if either: (1) it has non-negative advantage, or (2) its average log-probability ratio is within threshold $\delta$. The rationale from the paper:

> Models benefit the most by learning from its own mistakes, whereas highly off-policy negative samples can be detrimental, potentially misleading or destabilizing the optimization process.

The one-sided nature makes intuitive sense: for positive-advantage samples, we want to reinforce the behavior regardless of how much the policy has drifted. But for negative-advantage samples, learning from highly off-policy data can be counterproductive—the policy has already moved away from these samples, suggesting they're no longer representative of what the current policy would produce.

A crucial detail from DeepSeek about the notation:

> Note that $\pi_{\text{old}}$ here denotes the sampling probability directly returned by the inference framework, thus the KL divergence between the old and current policy accounts for both sources of off-policyness mentioned above.

In our three-policy notation, DeepSeek's $\pi_{\text{old}}$ is our $\mu$—the inference engine's distribution. So OPSM computes the divergence between the current policy $\pi$ and the inference distribution $\mu$, capturing both training-inference mismatch and policy staleness in a single ratio. This differs from the TIS/MIS correction between $\pi_{\text{old}}$ and $\mu$, which targets training-inference mismatch alone. Of course, nothing stops us from applying TIS/MIS to $\pi$ and $\mu$ if we wanted both sources handled together.

## opsm is geometric sequence masking

Let's look more carefully at OPSM's threshold term: $\frac{1}{T}\sum_t \log \frac{\mu}{\pi}$. This has the same form as the geometric mean we defined earlier—an average of per-token log-ratios—but with different policies. Earlier we used $\pi_{\text{old}}/\mu$ to correct for training-inference mismatch; OPSM uses $\mu/\pi$, or equivalently the reciprocal $\pi/\mu$.

We can define a geometric mean for this ratio too:

$$
\rho_{\text{geo}}^{\text{OPSM}} = \exp\left(\frac{1}{T}\sum_{t=0}^{T-1} \log \frac{\pi(y_t|x, y_{<t})}{\mu(y_t|x, y_{<t})}\right)
$$

With this definition, OPSM's threshold term $\frac{1}{T}\sum_t \log \frac{\mu}{\pi}$ is simply the negative: $-\log \rho_{\text{geo}}^{\text{OPSM}}$.

So OPSM's condition "$\frac{1}{T}\sum_t \log \frac{\mu}{\pi} > \delta$" becomes:

$$
-\log \rho_{\text{geo}}^{\text{OPSM}} > \delta \quad \Leftrightarrow \quad \rho_{\text{geo}}^{\text{OPSM}} < e^{-\delta}
$$

OPSM is a one-sided geometric sequence mask. It drops sequences where the current policy assigns significantly lower probability than the sampling distribution—where $\rho_{\text{geo}}^{\text{OPSM}}$ falls below the threshold $e^{-\delta}$.

| Aspect | Training-Inference Geo-Mask | DeepSeek OPSM |
|--------|------------------------|---------------|
| Direction | Two-sided ($C_{\min} \leq \rho_{\text{geo}}^{\text{TI}} \leq C_{\max}$) | One-sided ($\rho_{\text{geo}}^{\text{OPSM}} \geq e^{-\delta}$) |
| Advantage condition | None | Only applies when $\hat{A} < 0$ |
| Ratio | $\frac{\pi_{\text{old}}}{\mu}$ (training-inference only) | $\frac{\pi}{\mu}$ (both sources) |

But how do these two geometric means relate to each other? We can actually decompose $\frac{\pi}{\mu}$, and the relationship becomes clear:

$$
\frac{\pi}{\mu} = \frac{\pi_{\text{old}}}{\mu} \times \frac{\pi}{\pi_{\text{old}}}
$$

The first term is the training–inference mismatch. The second is policy staleness (how much $\pi$ has drifted from $\pi_{\text{old}}$ during gradient updates). OPSM still retains PPO's clipped surrogate objective on the $\frac{\pi}{\pi_{\text{old}}}$ importance ratio in GRPO, but complements it with a sequence-level mask which provides a hard safety valve when the effective off-policy-ness becomes too large.

My guess is that hard trust region enforcement becomes especially important as trajectories grow longer with agentic work and complex reasoning problems: even if per-token drift is modest, the tail behavior of updates can become messy, and keeping the average per-token divergence in a controlled range helps stability.

Applying this decomposition to $\rho_{\text{geo}}^{\text{OPSM}}$, we can rewrite OPSM's KL-threshold condition as:

$$
\frac{1}{T}\sum_t \log \frac{\mu}{\pi} = -\frac{1}{T}\sum_t \log \frac{\pi}{\mu} = -\frac{1}{T}\sum_t \log \left(\frac{\pi_{\text{old}}}{\mu} \cdot \frac{\pi}{\pi_{\text{old}}}\right)
$$

$$
 = - \left ( \underbrace{\frac{1}{T} \sum_{t} \log \frac{\pi_{\text{old}}}{\mu}}_{\log \rho_{\text{geo}}^{\text{TI}}} + \underbrace{\frac{1}{T} \sum_{t} \log \frac{\pi}{\pi_{\text{old}}}}_{\text{staleness term}} \right ) > \delta
$$


The first term, $\log \rho_{\text{geo}}^{\text{TI}}$, is exactly what we computed for training-inference correction earlier. OPSM's geometric mean is the training-inference geometric mean plus a staleness term that captures how much $\pi$ has drifted from $\pi_{\text{old}}$.

This decomposition clarifies what OPSM is really asking: "is this negative sample still representative of what the current policy might produce?" A sample fails this test if *either* the inference engine generated something the training model wouldn't have (large TI mismatch) *or* the policy has already moved away during optimization (staleness). In both cases, spending gradient signal to push down on it further is wasteful at best, destabilizing at worst. The single threshold $\delta$ acts as a joint budget for both sources of drift.

The decomposition is also useful for implementation: the training-inference term only needs to be computed once per rollout (it doesn't depend on $\pi$), while the staleness term changes with each gradient step.

If you already have geometric masking for training-inference mismatch, extending to OPSM means adding the staleness term:

```python
# From training-inference geo-mask: log ρ_geo^TI = (1/T) Σ log(π_old / μ)
# Computed once per rollout
ti_geo_mean = geo_mean_log_ratio

# Staleness term: (1/T) Σ log(π / π_old)
# Computed each gradient step as π updates
staleness_log_ratio = (per_token_logps - old_per_token_logps) * mask
staleness_geo_mean = staleness_log_ratio.sum(dim=-1) / seq_lengths

# OPSM's ratio: log ρ_geo^OPSM = log ρ_geo^TI + staleness
opsm_geo_mean = ti_geo_mean + staleness_geo_mean

# OPSM mask: keep if positive advantage OR low divergence
# Condition: -log(ρ_geo^OPSM) > δ  ⟺  log(ρ_geo^OPSM) < -δ
is_neg_adv = advantages < 0
is_high_divergence = opsm_geo_mean < -delta
opsm_mask = ~(is_neg_adv & is_high_divergence)
```

## a different view

The framing in this post treats the training-inference mismatch as a static numerical issue to be corrected. [Recent work from Yaxiang Zhang and Yingru Li](https://yingru.notion.site/Beyond-Precision-Why-Training-Inference-Mismatch-is-an-Optimization-Problem-and-How-Simple-LR-Sched-2d9211a558b780f1a710f99dbdc403d3) challenges this view, arguing that mismatch is better understood as a dynamic optimization problem coupled with gradient noise, and that targeted learning rate scheduling may be more effective than importance sampling corrections. Still too early to tell how this approach will adopt to a larger-scale, but it's an interesting alternative framing.

## references

Everything I've come to understand about Geometric Sequence Masking, and most of my understanding of importance sampling in this context, comes from Yingru Li's posts:

- [RL Collapse Series](https://richardli.xyz/post/rl-collapse-part1/)
- [When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda)
- [Mathematical Formulations of Rollout Correction Methods in verl](https://verl.readthedocs.io/en/latest/algo/rollout_corr_math.html)

Further reading and implementations:

- Fengyao's post on [off-policy RL in disaggregated systems](https://fengyao.notion.site/off-policy-rl)
- Ant Group's post on [IcePop for MoE training](https://ringtech.notion.site/icepop)
- [DeepSeek V3.2 technical report](https://arxiv.org/abs/2512.02556)
- [Stabilizing RL with LLMs: Formulation and Practices](https://arxiv.org/abs/2512.01374)

If you're interested in how OPSM fits into a post-training framework, have a look at [this TRL PR](https://github.com/huggingface/trl/pull/4891). For TIS/MIS specifically: [sequence-level MIS/TIS](https://github.com/huggingface/trl/pull/4530), [token-level TIS](https://github.com/huggingface/trl/pull/3867).
