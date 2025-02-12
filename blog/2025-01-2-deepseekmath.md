---
layout: post
title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
categories: []
year: 2024
type: paper
---

This reads like a classic foundational model paper, I've missed these. We've got some data curation tid bits, SFT details, and
RL, all heavily detailed and rigorously evaluated. With the answers in hand now that we've got R1, DeepSeekMath Corpus and GRPO are definitely the things to focus on in this paper.

To achieve this, we create the DeepSeek-
Math Corpus, a large-scale high-quality pre-training corpus comprising 120B math tokens. This
dataset is extracted from the Common Crawl (CC) using a fastText-based classifier

DeepSeekMath-Base is initialized with DeepSeek-Coder-Base-v1.5 7B (Guo et al., 2024), as
we notice that starting from a code training model is a better choice compared to a general
LLM. 

Our research provides compelling evidence that the publicly accessible Common Crawl
data contains valuable information for mathematical purposes. By implementing a metic-
ulously designed data selection pipeline, we successfully construct the DeepSeekMath
Corpus, a high-quality dataset of 120B tokens from web pages filtered for mathemati-
cal content, which is almost 7 times the size of the math web pages used by Minerva
(Lewkowycz et al., 2022a) and 9 times the size of the recently released OpenWebMath
(Paster et al., 2023).

### Dataset
DeepSeek provide a detailed description of how their dataset is curated. This is rare these days, very impressive. First, we choose OpenWebMath (Paster et al., 2023), a collection of high-quality mathematical
web texts, as our initial seed corpus. Using this corpus, we train a fastText model (Joulin et al.,
2016) to recall more OpenWebMath-like mathematical web pages. Specifically, we randomly
select 500,000 data points from the seed corpus as positive training examples and another
500,000 web pages from Common Crawl as negative ones. We employ an open-source library1
for training, configuring the vector dimension to 256, learning rate to 0.1, the maximum length of word n-gram to 3, the minimum number of word occurrences to 3, and the number of training epochs to 3. To reduce the size of the original Common Crawl, we employ URL-based
deduplication and near-deduplication techniques, resulting in 40B HTML web pages. We then
recall mathematical web pages from deduplicated Common Crawl with the fastText model.
To filter out low-quality mathematical content, we rank the collected pages according to their
scores predicted by the fastText model, and only preserve the top-ranking ones. The volume
of data preserved is assessed through pre-training experiments on the top 40B, 80B, 120B, and
160B tokens. In the first iteration, we choose to keep the top 40B tokens.

We run pre-training experiments to investigate how the DeepSeekMath Corpus is compared
with the recently released math-training corpora:
• MathPile (Wang et al., 2023c): a multi-source corpus (8.9B tokens) aggregated from
textbooks, Wikipedia, ProofWiki, CommonCrawl, StackExchange, and arXiv, with the
majority (over 85%) sourced from arXiv;
• OpenWebMath (Paster et al., 2023): CommonCrawl data filtered for mathematical content,
totaling 13.6B tokens;
• Proof-Pile-2 (Azerbayev et al., 2023): a mathematical corpus consisting of OpenWeb-
Math, AlgebraicStack (10.3B tokens of mathematical code), and arXiv papers (28.0B to-
kens).

A 1.3B DeepSeek LLM trained on the DeepSeekMath Corpus significantly outperform those trained on other
math-related corpus listed above. The results are displayed in this image ![](/images/dsmathcorpus.png)

DeepSeekMath Corpus totals 120B tokens, several times larger than existing mathematical corpora. As proven
it also is of higher quality than existing corpus.

## DeepSeekMath Base Model
we introduce DeepSeekMath-Base 7B, a base model with strong reasoning
abilities, especially in mathematics. Our model is initialized with DeepSeek-Coder-Base-v1.5 7B  and trained for 500B tokens. The distribution of the data is as follows: 56%
is from the DeepSeekMath Corpus, 4% from AlgebraicStack, 10% from arXiv, 20% is Github
code, and the remaining 10% is natural language data from Common Crawl in both English and
Chinese.

DeepSeekMath-Base 7B comes with significant improvement in mathematical problem solving (GSM8K, MATH, OCW, SAT), problem solving with tool use (GSM8K+Python, MATH+Python) and formal mathematics (miniF2F), while also improving over its base model in language understanding and reasoning - illustrating the positive impact of math training on adjacent tasks. Additionally, by including code tokens for continual training, DeepSeekMath-Base 7B effectively maintains the performance of DeepSeek-Coder-Base-v1.5 on the two coding benchmarks.

## Supervised Fine Tuning

In this section, we introduce DeepSeekMath-Instruct 7B which undergoes mathematical instruc-
tion tuning based on DeepSeekMath-Base.

We construct a mathematical instruction-tuning dataset covering English and Chinese problems
from different mathematical fields and of varying complexity levels: problems are paired with
solutions in chain-of-thought (CoT) (Wei et al., 2022), program-of-thought (PoT) (Chen et al.,
2022; Gao et al., 2023), and tool-integrated reasoning format (Gou et al., 2023). The total number
of training examples is 776K.

As shown in Table 5, under the evaluation setting where tool use is disallowed, DeepSeekMath-
Instruct 7B demonstrates strong performance of step-by-step reasoning. Notably, on the
competition-level MATH dataset, our model surpasses all open-source models and the ma-
jority of proprietary models (e.g., Inflection-2 and Gemini Pro) by at least 9% absolute. This
is true even for models that are substantially larger (e.g., Qwen 72B) or have been specifi-
cally enhanced through math-focused reinforcement learning (e.g., WizardMath-v1.1 7B). While
DeepSeekMath-Instruct rivals the Chinese proprietary models GLM-4 and Baichuan-3 on MATH,
it still underperforms GPT-4 and Gemini Ultra.

Under the evaluation setting where models are allowed to integrate natural language rea-
soning and program-based tool use for problem solving, DeepSeekMath-Instruct 7B approaches
an accuracy of 60% on MATH, surpassing all existing open-source models.




## Reinforcement Learning
Now we've got to the good stuff. Group Relative Policy Optimization. Here's a quick primer on PPO before we get started

## ppo crash course
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

It may not seem natural at first to apply PPO in the context of a language model, but I assure you it's no different. 

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