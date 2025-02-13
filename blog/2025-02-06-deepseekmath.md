---
layout: post
title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
categories: []
year: 2024
type: paper
---

Late on the ball again... let's see what all the r1 hype is about.

DeepSeekMath presents a large-scale mathematical pretraining dataset and model, structured around dataset curation, supervised fine-tuning (SFT), and reinforcement learning (RL). The paper provides a meticulous breakdown of how the dataset was constructed, why certain architectural choices were made, and how the resulting model performs across key benchmarks. 

The dataset is constructed from Common Crawl and refined through a fastText-based classifier, ultimately yielding 120 billion tokens. This is an order of magnitude larger than existing math corpora—7x the size of Minerva’s dataset, 9x OpenWebMath. The data pipeline is designed to maximize relevance and quality, starting with OpenWebMath as a seed corpus and using a classifier trained with 500,000 positive examples from OpenWebMath and 500,000 negative examples from Common Crawl. 

Deduplication techniques are employed to reduce noise, trimming Common Crawl down to 40 billion HTML pages. The classifier then selects mathematical content, ranking it by relevance and preserving the highest-scoring entries. The final dataset selection is guided by pretraining experiments, assessing performance at different token thresholds (40B, 80B, 120B, 160B).

### DeepSeekMath-Base 7B: Pretraining and Model Structure

DeepSeekMath-Base is initialized with DeepSeek-Coder-Base-v1.5 (a 7B parameter model originally trained for code) and further trained on 500 billion tokens, distributed as follows:
- 56% DeepSeekMath Corpus
- 4% AlgebraicStack
- 10% arXiv
- 20% GitHub code
- 10% Common Crawl (English & Chinese)

Starting from a code-focused base model is a deliberate choice—empirical evidence suggests that models with coding exposure generalize better to mathematical reasoning tasks than those trained purely on natural language. This pretraining setup results in significant gains on GSM8K, MATH, OCW, and SAT benchmarks, while also maintaining strong coding proficiency. Unlike models that degrade in one domain when fine-tuned for another, DeepSeekMath-Base effectively preserves its coding capabilities.

### DeepSeekMath-Instruct 7B: Supervised Fine-tuning

To further improve mathematical reasoning, DeepSeekMath-Instruct 7B is fine-tuned with 776,000 instruction-style problems, covering:
- Chain-of-Thought (CoT) for step-by-step reasoning
- Program-of-Thought (PoT) for algorithmic solutions
- Tool-integrated reasoning for leveraging external computational resources

Without tool assistance, DeepSeekMath-Instruct 7B outperforms all open-source models, including Inflection-2, Gemini Pro, and Qwen 72B, surpassing them by at least 9% absolute on the MATH benchmark. Even against proprietary models, it remains competitive, though GPT-4 and Gemini Ultra still hold the lead.

When allowed to incorporate external tools (e.g., symbolic computation, programming), DeepSeekMath-Instruct 7B reaches ~60% accuracy on MATH, establishing itself as the strongest open-source model for mathematical reasoning.

## Reinforcement Learning
So far things are pretty standard, a new open dataset -> a better model. Great stuff, and in practice a lot of engineering behind this of course but in theory pretty standard. But now we get to what I've been actually looking forward to in this paper - Group Relative Policy Optimization. However, before we do so, I'd like to provide a quick primer on ppo for those who are unfamiliar.

## *ppo crash course*
Reinforcement learning has long struggled with the challenge of stable and efficient policy optimization. Early policy gradient methods, while theoretically sound, were notorious for their instability—small changes in policy could lead to catastrophic drops in performance, making training unpredictable. Standard approaches, such as vanilla policy gradients, suffered from high variance and lacked a mechanism to prevent overly aggressive updates.

Trust Region Policy Optimization (TRPO) was a big deal when it came out. It tackled one of the biggest problems in reinforcement learning: instability in policy gradient methods. The idea was simple—if we update our policy, we should ensure that we don’t make such a large step that we completely destroy performance. TRPO enforced this by constraining the KL-divergence between the old and new policies, keeping things stable. By maintaining updates within a "trust region," TRPO significantly improved training reliability, making it a go-to choice for many reinforcement learning applications.

But TRPO had its downsides. It was a second-order method, meaning it required expensive computations, including solving a constrained optimization problem at every update step. This made it computationally intensive and cumbersome to implement in practice. Enter Proximal Policy Optimization (PPO), which asked the same question—how can we take the largest possible policy improvement step while ensuring stability?—but answered it with a much simpler first-order approach. By replacing TRPO’s complex optimization constraints with a clipped objective function, PPO retained stability while being far easier to implement and scale.

At its core, PPO follows a straightforward loop:

1. Collect a dataset of trajectories by running the current policy in the environment.
2. Score these trajectories using a reward model.
3. Compute General Advantage Estimates (GAE) based on the critic’s value function.
4. Update the policy using stochastic gradient descent (SGD) and the Adam optimizer, maximizing the PPO-Clip objective.
5. Repeat for a long time.
6. Profit.

PPO brings two key ingredients to the table: **policy optimization** and **advantage estimation**. The synergy between these is what makes PPO work.

- **Policy Optimization:** The agent collects experience by interacting with the environment. It then updates its policy to maximize expected cumulative reward while ensuring updates stay within a safe range.
- **Value Function Estimation:** The value function estimates the expected future rewards, helping calculate an advantage signal to guide policy updates.

### The PPO-Clip Objective

PPO introduces a clipped objective function to ensure stable updates:

$$
L(s, a, \theta_{k}, \theta) = \min \left( r A, g(\epsilon, A) \right)
$$

where

$$
r = \frac{\pi_\theta(a|s)}{\pi_{\theta_k}(a|s)}
$$

is the probability ratio between the new and old policy, and $A$ is the estimated advantage.

The function $g(\epsilon, A)$ is defined as:

$$
g(\epsilon, A) =
\begin{cases} 
(1 + \epsilon) A & A \geq 0 \\
(1 - \epsilon) A & A < 0.
\end{cases}
$$

### Why Clipping Matters

The clipping mechanism is what makes PPO robust. It prevents excessive policy updates, ensuring we don’t overcorrect in either direction.

**Case 1: Positive Advantage ( $A > 0$ )**
- The action was better than expected.
- We want to increase the probability of taking this action.
- If $ r > 1$, the new policy is already favoring this action.
- Clipping prevents excessive updates in this direction.

**Case 2: Negative Advantage ($A < 0$)**
- The action was worse than expected.
- We want to decrease its probability.
- If $ r < 1$, the policy is already discouraging it.
- Clipping ensures we don’t over-penalize it.

Essentially, PPO encourages good actions but prevents excessive reinforcement, and discourages bad actions but prevents complete suppression. This keeps training stable.

### General Advantage Estimation

At this point, you might be wondering—what exactly is "advantage"? Advantage measures how much better an action was compared to the expected value of the state:

$$
A(s_t, a_t) = Q(s_t, a_t) - V(s_t)
$$

where $Q(s_t, a_t)$ is the expected cumulative reward for taking an action $a_t$ in state $s_t$, and $V(s_t)$ is the expected cumulative reward of the average action the policy takes in state $s_t$. Naturally, we can either use the reward for the full trajectory, called **Monte-Carlo**, which results in a low bias but high variance estimation. Alternatively, we can bootstrap, estimating the value fo future states using a single step lookahead, without waiting for full Monte-Carlo returns:

$$
A_t = r_t + \lambda V(s_{t+1}) - V(s_t).
$$

This is sample efficient but it introduces high bias. General Advantage Estimate (GAE) interpolates between these two approaches, introducing a weighted sum of k-step temporal differences, controlled by hyperparameter $\gamma$, to balance the bias-variance tradeoff. It might sound complicated but we're just calculating the temporal difference at $K$ steps and taking the weighted sum of these errors:

$$
A_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\lambda)^l \delta_{t+l}
$$

where $\delta_{t+l}$ is the temporal difference:

$$
\delta_{t+l} = V(s_{t+1}) - V(s_t).
$$

GAE integrates short-term and long-term information, balancing bias and variance in advantage estimation. When $\lambda = 0$, it collapses to one-step TD, relying solely on immediate estimates. When $\lambda = 1$, it becomes Monte Carlo estimation, using full trajectory returns. 

To bring this together—GAE determines advantage by observing how the critic network's value estimate evolves over the next $K$ steps. A positive advantage means future states are expected to be more valuable, reinforcing the action taken. By blending MC and TD, GAE smooths advantage estimation, reducing variance while keeping bias in check, which helps stabilize PPO training. It effectively combines short-term corrections with long-term trends, ensuring more reliable updates. 

If this still feels abstract, consider it this way: we’re looking a few steps ahead, estimating the value at each state, and comparing those values to judge whether the action taken in the initial state was actually beneficial.

For those who prefer code over formulas

![](/images/gaecode.png)


### concluding thoughts

PPO works because it updates the policy efficiently while keeping changes constrained. The clipped objective ensures stability, while GAE provides a reliable advantage estimate. A lot of the tricks of PPO boil down simplicity and stability, taking small updates steps within our trusted region with a smooth advantage estimation.  If you take one thing away from this, it's that PPO is just doing **trust region optimization, but without all the second-order math.** 

---

It may not seem natural at first to apply PPO in the context of a language model, but I assure you it hardly requires work. 

- **Policy** $\pi_\theta$: the LLM that has been pre-trained / SFT’ed
- **Reward model** $R_\phi$: a trained and frozen network that provides scalar reward given complete response to a prompt
- **Critic** ($V_\gamma$): also known as value function, which is a learnable network that takes in partial response to a prompt and predicts the scalar reward.

This formulation enabled the application of PPO on LLMs.

One of the downsides of PPO is the need for a value function, or critic network, to be trained alongside the policy model, and to avoid over optimization of the reward model (especially true for LLMs) the standard approach is to add a per-token KL penalty from a reference model in the reward at each token. The regularizer is often the initial SFT model. 

As the value function employed in PPO is typically another model of comparable size as
the policy model, it brings a substantial memory and computational burden. In standard PPO, the value function serves as a baseline for advantage estimation, helping to reduce variance in training. The advantage function measures how much better or worse a particular action is compared to expected performance, and the value function plays a crucial role in stabilizing this calculation. However, in traditional reinforcement learning settings like robotics or games, rewards are typically assigned at every step, making it straightforward to train a value function that predicts future rewards for each action. In contrast, reinforcement learning for LLMs operates differently. Instead of assigning rewards to every token, the reward model typically provides a single score for the entire generated sequence, meaning only the final token explicitly receives a reward. This creates a fundamental challenge: PPO relies on per-token advantage estimates, but since the value function is trained to predict future rewards, it has no clear target for intermediate tokens. Training a value model in this context is difficult because it must infer how the final reward should be distributed across the sequence, leading to high variance and instability. As a result, using a standard PPO approach with LLMs can be problematic. The value model struggles to learn meaningful token-wise predictions, which can degrade the quality of advantage estimation and make training unreliable. 



## ablations / studies 

Code training also improves mathematical reasoning without tool use. 

However, combining code tokens and math tokens for one-stage training com-
promises mathematical reasoning without tool use. One conjecture is that DeepSeek-LLM 1.3B,
due to its limited scale, lacks the capacity to fully assimilate both code and mathematical data
simultaneously.

ArXiv papers are commonly included as a component of math pre-training data (Azerbayev
et al., 2023; Lewkowycz et al., 2022a; Polu and Sutskever, 2020; Wang et al., 2023c). However, detailed analysis regarding their impact on mathematical reasoning has not been extensively
conducted. Perhaps counter-intuitively, according to our experiments, arXiv papers seem
ineffective in improving mathematical reasoning.

When trained on a arXiv-only corpus, both models dis-
play no notable improvements or even deterioration across various mathematical benchmarks of
different complexities employed in this study. 

However, this conclusion has its limitations and should be taken with a grain of salt. We
have not yet studied:
- The impact of arXiv tokens on specific math-related tasks not included in this research,
such as informalization of theorems which is to convert formal statements or proofs to
their informal versions;
- The effect of arXiv tokens when combined with other types of data;
- Whether the benefits of arXiv papers would manifest themselves at a larger model scale.