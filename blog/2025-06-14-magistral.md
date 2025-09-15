---
layout: post
title: "Magistral"
categories: []
year: 2025
type: paper
author: Rastogi
exturl: https://arxiv.org/abs/2506.10910
---

Mistral has recently released its new line of reasoning-focused models, Magistral. Unfortunately the hype for mistral has died down in the past year. They've fallen behind considerably, with Chinese labs taking their place as the non-us competition. 

Magistral is built upon the Mistral Small and Mistral Medium foundation models, applying a Reinforcement Learning from Verifier Feedback (RLVF) framework to enhance their reasoning capabilities.

### GRPO-ish

In line with a major trend in reinforcement learning over the past year, Mistral has adopted a variant of GRPO as its core RL algorithm. The rapid adoption and iteration on GRPO across the research community have been remarkable, and Mistral's implementation adds another data point to this cycle.

Mistral’s GRPO implementation incorporates several key modifications, many of which draw inspiration from recent papers like DAPO and Dr GRPO:

1.  **KL Divergence Removal:** Mistral's version omits the KL divergence term, a topic of ongoing debate. While some research suggests KL divergence is crucial for preventing policy entropy collapse, others have sought alternatives. For instance, the ProRL paper retained KL divergence but periodically reset the reference policy to an on-policy checkpoint during training. Mistral's choice to remove it entirely simplifies the objective.

2.  **Loss Normalization (from DAPO):** The loss is normalized by the total length of the generated text. This is a common technique designed to prevent the model from developing a bias toward unnecessarily short or long responses.

3.  **Modified Advantage Normalization:** This is a noteworthy change. The original Dr GRPO paper proposed dividing advantages by their standard deviation ($\sigma$). However, this can introduce a "difficulty bias," as questions that are either too easy or too hard (resulting in low variance in rewards) are given disproportionately high weights during policy updates. Mistral refines this by normalizing advantages at the minibatch level:
    $$A_{\text{norm}} = \frac{A - \mu_A}{\sigma_A + \epsilon}$$
    Here, the mean ($\mu_A$) and standard deviation ($\sigma_A$) are calculated across all advantages within a given minibatch, ensuring more stable and balanced updates.

4.  **Upper-Bound Clipping (from DAPO):** The implementation uses "Clip-Higher," setting the upper epsilon clipping value between 0.26 and 0.28.

5.  **Dynamic Batch Filtering (from DAPO):** To improve sample efficiency, training batches are formed by filtering out generation groups that have zero advantage.

The heavy reliance on techniques from the DAPO paper (KL removal, loss normalization, clipping, and dynamic sampling) suggests a rapid and recent development cycle, likely within the last couple of months.

### Rewards

Mistral's reward shaping is straightforward but prescriptive. The model is rewarded for using specific formatting, such as `<think>` tags for chain-of-thought, `\boxed` for mathematical answers, and markdown for code blocks. The reward logic is strict:

> Failure to meet any of these conditions results in a reward of 0, and the response will not be graded further. Otherwise, the response gets a reward of 0.1 and proceeds to grading.

This all-or-nothing formatting gate appears to have caused some "reward hacking." I have observed instances where the model inappropriately uses mathematical formatting for standard questions, likely because the RL process over-indexed on this signal. A more robust approach might involve using a detailed reward rubric rather than enforcing specific syntax, as RL-trained behaviors often bleed into unintended domains. Additionally, the model incorporates the length penalty reward signal from DAPO.

### RL Infra

Mistral developed a distributed, asynchronous RL training system with three standard components: **Generators**, **Trainers**, and **Verifiers**.

The primary bottleneck in such systems is the **Generators** (the LLMs performing rollouts), as generation time varies significantly—the longest completions can take five times longer than the shortest. The key challenge is to keep the generators as on-policy as possible by updating their weights frequently while maximizing hardware utilization.

A simple sequential process (generate -> wait -> update) would be highly inefficient. Instead, Mistral’s generators run asynchronously, continuously producing rollouts at maximum throughput. Solutions are gathered, verified, and used to update the **Trainers**. Once a trainer completes a gradient update, the new weights are broadcast to the generators via NCCL. Interestingly, this update happens mid-generation, without discarding in-flight sequences. This means the KV cache becomes temporarily outdated relative to the new model weights, a trade-off Mistral made for higher system throughput.

### Training and Results

The experiments were designed to answer two questions:
1.  How effective is pure reinforcement learning on a large base model?
2.  How can one create the strongest possible lightweight model using a powerful teacher?

**Magistral Medium**, trained from Mistral Medium, addresses the first question. **Magistral Small** addresses the second, having been trained with SFT traces derived from Magistral Medium.

The training of Magistral Medium proceeded in three stages, which included:
* **Curriculum Learning:** The difficulty of the training data was progressively increased by filtering out easier examples as the model's performance improved.
* **Dynamic Length Penalty:** The length penalty was adjusted throughout training to encourage the model to generate progressively longer and more complex reasoning chains without stagnating.