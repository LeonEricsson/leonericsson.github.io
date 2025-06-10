---
layout: post
title: "The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models"
categories: []
year: 2025
type: paper
---
The authors explore four model families across eleven popular open-source checkpoints. Starting from these base models, they apply three reinforcement-learning algorithms—**GRPO**, **REINFORCE**, and **PRIME**—with the KL-divergence coefficient fixed at 0.

A striking pattern emerges everywhere: **policy entropy plummets almost immediately and then continues its monotonic slide to ≈ 0, while validation performance rises in near perfect anti-phase**. After just *200 gradient steps* (≈ 1∕12 of total training), entropy has already fallen by **73 %**, and *76 %* of the eventual performance gain has been realized.

Empirically, the trade-off is well-captured by the exponential fit

$$
R \;=\; -a\,e^{H} + b,
$$

where $R$ is validation score and $H$ is entropy. The figure below (omitted here) shows the tight fit across models.

Using coefficients $a, b$ estimated from the *first 36* steps, the authors predict performance for the next 200 steps with an RMSE ≈ 1 %. Setting $H = 0$ also gives an *upper bound* on achievable validation performance.

#### Interpreting $a$ and $b$

Both coefficients are **algorithm-agnostic**: for a fixed model size they stay constant regardless of the RL method, suggesting they encode intrinsic properties of the policy prior and data distribution.

$$
\frac{dR}{dH} = -a\,e^{H} \;\;\Longrightarrow\;\; a
$$

is literally the *conversion rate* between entropy and downstream reward, while $-a + b$ is the theoretical performance ceiling. Empirically, $a$ and $b$ vary *log-linearly* with model size—so once you have them for small models, you can extrapolate to larger ones and **predict their final RL-tuned performance without training them**.

A common concern is that RL can only *elicit* behaviors latent in the pre-trained distribution, never exceed them. The data partially corroborate that fear: once entropy has collapsed, the ceiling is both real and predictable. Crucially, the authors argue the culprit is not RL per se but the **entropy dynamics of large language models**—strong priors narrow the output distribution, limiting exploration.

### The Dynamics of Policy Entropy

Given that entropy collapse appears to be a primary obstacle for scaling RL in language model reasoning, it's critical to understand *when* entropy will increase or decrease. The paper provides a mathematical derivation for the change in entropy under policy gradient algorithms, starting from the fact that an LLM is a softmax policy.

The core finding is that a strong positive correlation between an action's probability under the current policy, $P(a)$, and its corresponding advantage value, $A(a)$, leads to a **decrease in policy entropy**. More formally, the change in entropy is determined by the covariance between the log-probability of an action (the model's confidence) and the advantage of that action (how good it was relative to the expected outcome).

-   A **positive covariance** decreases entropy. This occurs during **exploitation**, where a high-probability action yields a high reward, or during **confirmation**, where a low-probability action results in a low reward. In both cases, the model's beliefs are reinforced, narrowing the policy.
-   A **negative covariance** increases entropy. This happens during **exploration**, where a low-probability action leads to a surprisingly high reward, or during **correction**, where a confident action results in a poor outcome. Both scenarios challenge the model's current beliefs, encouraging it to broaden its policy.

In practice, RL for reasoning tasks is dominated by **exploitation**. The model rapidly identifies "easy wins"—high-probability actions that yield high advantages—causing the policy entropy to collapse.

### Entropy Control via Covariance Regularization

To counteract this dynamic, the authors propose controlling policy updates through regularization. Standard approaches like adding an entropy bonus or a reference-KL penalty to the loss function proved ineffective. Instead, the authors developed a filtering-based approach motivated by the observation that a small fraction of tokens contributes disproportionately high covariance.

They propose two methods:

1.  **Clip-Cov**: After calculating the covariance of each token in a batch, this method clips a small fraction of the highest-covariance tokens from the policy gradient update by zeroing out their gradients.
2.  **KL-Cov**: A simpler approach that first ranks tokens by their covariance and then applies a KL penalty only to the top-$k$ proportion.

Both methods are designed to mitigate the entropy collapse driven by high-covariance exploitation events, thereby preserving the model's ability to explore and improve performance.
