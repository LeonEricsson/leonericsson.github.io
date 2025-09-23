---
layout: post
title: "LongCat-Flash"
categories: []
year: 2025
type: paper
author: LongCat Team
exturl: https://arxiv.org/abs/2509.01322
---

It's becoming a pattern: another chinese lab, seemingly appears out of nowhere and releases a massive +500B parameter model. Just a year ago, this scale of open release would have been unheard of. Hot on the heels of the base model, LongCat-Flash, they've already released LongCat-Flash-Thinking, a dedicated reasoning model. But before we get into '-Thinking', I figure we want to take a close look at the base model that underpins it. It seems I completely missed, or ignored, the talk about this model on X when it was released a few weeks ago, but writing this after I've read the paper I can tell you this is an insane release, wow, I'm thoroughly impressed. 

LongCat-Flash is a 560B MoE with a dynamic computational budget allocation that allows it to activate between 18-31B per token depending on contextual demands. They report that the model is trained on over 20 trillion tokens within 30 days. This is an impressive feat of engineering, especially for a team that seems to have limited experience in the field. Let's do some quick napkin math to get a sense of the scale.

We can derive the popular `6 * num_tokens * num_active_parameters` rule of thumb for training FLOPs from first principles. For a transformer layer, the cost is dominated by the matrix multiplications in the attention projections and the FFNs.

* The **attention block** consists of four projections (for Q, K, V, and Output), which for a forward and backward pass require approximately `24 * B * S * D^2` FLOPs, where `B` is batch size, `S` is sequence length, and `D` is the hidden dimension.
* An **MoE FFN block** activating `E` experts, each with an intermediate size of `F`, requires about `12 * B * S * D * E * F` FLOPs.

Combining these for `L` layers, plus the embedding/unembedding layers, gives the total FLOPs. A simpler way is to sum the active parameters and multiply by `6 * num_tokens`. The total number of active parameters in a forward pass is the sum of the non-MoE parameters (attention, layer norms, embeddings) and the parameters from the activated experts. For a well-designed MoE model, this is dominated by the expert parameters.

Because LongCat-Flash has a dynamic forward pass, the exact compute is variable, but the paper states they activate **27B parameters on average**. Using this, we can estimate the total training FLOPs:

$$6 \times (27 \times 10^9 \text{ params}) \times (20 \times 10^{12} \text{ tokens}) = 3.24 \times 10^{24} \text{ FLOPs}$$

The paper specifically mentions using H800s for their inference stack, so we can reasonably assume they were used for training as well. An NVIDIA H800 delivers around 989 TFLOP/s for BF16, but achieving high Model FLOPs Utilization (MFU) is challenging. Assuming a generous **50% MFU** (495 TFLOP/s), the number of GPUs required to complete training in 30 days would be:

$$\frac{3.24 \times 10^{24} \text{ FLOPs}}{495 \times 10^{12} \text{ FLOP/s} \times (30 \times 24 \times 3600 \text{ s})} \approx 2525 \text{ GPUs}$$

This is a lower bound, as we've ignored some computations and assumed a high MFU. The paper mentions using **"tens of thousands of accelerators,"** which suggests either a much lower MFU, a much shorter training run than 30 days, or simply having a massive cluster at their disposal. Regardless, a training run of this magnitude from a relatively unknown commercial lab is a testament to the rapid democratization of large-scale AI

### arch
this is a non standard MoE with a couple of modifications that are are novel in a model of this capacity.

#### zero-computation experts
Naturally, some tokens are "harder" to predict then others. The paper argues that such tokens should demand more resources. One can argue for this case by looking at the success of speculative decoding, where a smaller draft model is able to accurately predict certain output sequences of larger models. We've seen similar analysis earlier establish the existence of *fork* tokens which very important for reasoning sequences. Anyway, motivated by this theory, LC introduces a dynamical resource allocation method in the MoE layer. This is achieved by introducing new experts in the MoE which don't perform any compute and simply just let the input pass through, LC introduces $Z$ of these *zero-compute* experts alongside $N$ normal FFN experts. This allows the model to choose how much compute to assign to a certain token. The MoE layer is formalized as follows:

Not all tokens are created equal; some are inherently "harder" to predict than others. The paper argues that these harder tokens should demand more computational resources. This idea is empirically supported by the success of speculative decoding, where a small draft model can accurately predict outputs for "easy" tokens from a much larger model.

Motivated by this, LongCat-Flash introduces a mechanism for dynamic computational allocation. This is achieved by adding $Z$ "zero-computation" experts to the standard pool of $N$ FFN experts within each MoE layer. These special experts perform no computation and simply pass the input through as an identity function ($E_i(x_t) = x_t$). This design allows the model to learn to allocate a variable number of FFN experts to each token based on its contextual importance. The MoE layer is formalized as follows:

$$
\begin{equation}
\begin{aligned}
\text{MoE}(x_t) &= \sum_{i=1}^{N+Z} g_i \, E_i(x_t), \\
g_i &=
\begin{cases}
R(x_t)_i, & \text{if } R(x_t)_i \in \text{TopK}\!\big(R(x_t)_i + b_i \,\big|\, 1 \leq i \leq N+Z, K\big), \\
0, & \text{otherwise},
\end{cases} \\
E_i(x_t) &=
\begin{cases}
\text{FFN}_i(x_t), & \text{if } 1 \leq i \leq N, \\
x_t, & \text{if } N < i \leq N+Z.
\end{cases}
\end{aligned}
\end{equation}
$$

where $g_i$ are the router gates for the top-K selected experts and $E_i(x_t)$ is the output of the $i$-th expert.

To guide the routing and ensure a stable computational budget, the model includes a bias term $b_i$ in the router logic, which is dynamically adjusted using a PID controller. This is a refinement of the aux-loss-free strategy seen in other models, designed to maintain a target average number of activated FFN experts. The result is a model that allocates its computational budget more effectively—achieving a lower validation loss than a baseline model with a fixed compute budget equivalent to LongCat-Flash's average. This confirms that it's beneficial to trade less compute on some tokens for more compute on others, and that the model can learn this allocation itself.

#### shortcut-connected moe
Expert parallelism (EP) is crucial for scaling MoE models, as it distributes experts across accelerators. However, the conventional execution paradigm imposes a sequential workflow: a collective communication operation (all-to-all) must route tokens to their designated experts *before* expert computation can begin. This communication latency creates a bottleneck that is difficult to hide with computation. While shared-expert architectures attempt to mitigate this by overlapping communication with the computation of a single shared expert, the computational window is often too small to be effective.

LongCat-Flash employs Shortcut-connected MoE (ScMoE) to address this limitation. The key idea behind ScMoE is to reorder the execution pipeline by feeding the MoE block with the output from an earlier part of the transformer layer (specifically, the output of the first Multi-head Latent Attention block). This creates a much larger computational window—the dense FFN from the preceding block—that can execute in parallel with the MoE's dispatch and combine communication phases. The paper validates that this architectural change is quality-neutral, showing nearly identical loss curves to a standard MoE, but provides substantial system-level efficiency gains for both training and inference. In practice, this design choice reduced the proportion of time spent on non-overlapping communication from 25.3% to just 8.4%.

The overall transformer block structure can be summarized as:
* A dense path: `MLA_1 -> FFN_1 -> MLA_2 -> FFN_2`
* A sparse path: The MoE block takes its input from the output of `MLA_1`.
* The final output is the sum of the dense path output (`FFN_2`) and the sparse path output (`MoE`).

The model uses MLA, consists of only 28 layers, and contains a staggering 512 FFN experts and 256 zero-computation experts per MoE block. The router activates 12 experts per token, with a target of activating ~8 FFN experts on average. This results in an incredibly high sparsity factor of 64 (an activation ratio of 1.56%), which is higher than I've ever seen before. You typically don't see this level of sparsity due to the infra complexity that comes with it, but from what I can gather ScMoe alleviates this making it possible.

### training

The training methodology also includes some interesting techniques.

LongCat-Flash uses **model growth initialization**, a strategy I haven't seen in a production model before. Instead of training the full 28-layer model from a random initialization, they first train a half-scale model (14 layers) on tens of billions of tokens. They then use a layer stacking technique to duplicate this pre-trained model, creating a 28-layer checkpoint that serves as the initialization for the full training run. The paper shows this leads to faster convergence and ultimately better performance compared to random initialization.

Another impressive engineering detail is the focus on **determinism**. The team developed custom, deterministic kernels for key operations like FlashAttention Gradients (FAG) and ScatterAdd that improve on existing deterministic implementations while maintaining high performance. This guarantees bitwise reproducibility of experiments, a critical feature for debugging and reliable research at scale. The recent interest in deterministic training makes it very cool to see this level of engineering detailed in a paper.

The paper continues with an in-depth discussion of their distributed training strategy and inference optimizations, including communication/computation overlapping, custom kernels, and speculative decoding. It's a dense and highly informative read that I recommend for anyone interested in the systems side of training and serving massive models.