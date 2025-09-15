---
layout: post
title: "Gated Attention"
categories: []
year: 2025
type: paper
---

Qwen3-Next is, according to the team themselves, a representation of the *next-generation architecture*, built as a hybrid attention model that alternates between global attention layers (in the form of **Gated Attention**) and linear attention layers (**Gated DeltaNet**). [DeltaNet](https://sustcsonglin.github.io/blog/2024/deltanet-1/) has been on my reading list for some time now, but linear attention is a whole field in its own right at this point so I've sort of put it off until I have time for a justified deeper dive. We'll get there. Hybrid architectures are slowly gaining popularity in the open space and I don't see this trend slowing down, it seems like a necessity towards longer contexts; the time of full attention timemixing is coming to an end. 

Anyway, I'm still putting off DeltaNet for now, instead I want to look at the gating mechanism that appears central to the Qwen3-Next design, applied to both of its attention modules. The team recently published an in-depth analysis motivating this choice, titled ["Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free"](https://www.google.com/search?q=https://arxiv.org/abs/2505.06708v1), which systematically explores different gating configurations within the standard attention block. Let's walk through their findings to understand what convinced them this component was next-generation worthy.

The paper's investigation focuses on applying various gating setups to a standard multi-head attention module. The gating mechanism is formalized as:

$$Y' = g(Y, X, W_\theta, \sigma) = Y \odot \sigma(XW_\theta)$$

where $Y$ is the input tensor to be modulated, $X$ is a separate input used to compute the gating scores (in this case, the pre-normalization hidden state), $W_\theta$ represents the learnable parameters of the gate, and $\sigma$ is an activation function, typically a sigmoid.

The authors exhaustively test gates at five distinct positions within the attention block: after the query, key, and value projections ($G_4$, $G_3$, $G_2$), after the Scaled Dot-Product Attention (SDPA) output ($G_1$), and after the final dense output layer ($G_5$). The experiments also cover different granularities (**head-specific** vs. **head-shared**) and application methods (a single scalar score per head vs. **elementwise** scores). 

The main results are compelling; the improvements aren't astounding but they are consistent and gating proves almost universally beneficial consistently improving perplexity and downstream benchmark scores. The optimal configuration is an **elementwise, head-specific, multiplicative sigmoid gate** applied directly to the SDPA output ($G_1$). Gating the value projection ($G_2$) is also highly effective, though slightly less so. Crucially, the experiments show that **head-specific** gating is vital; sharing gates across heads diminishes the performance gains. Another key finding is the significant improvement in training stability—gating allows for larger learning rates and batch sizes while mitigating loss spikes.

In practice, this is very straight forward to implement. For example, the optimal configuration of element-wise, head-specific, multiplicative sigmoid gate applied directly to the SDPA output looks like:

```python
def forward(
        self,
        x: Float[Tensor, "batch seq d_model"],
        mask: Float[Tensor, "1 1 seq seq"],
        pos_info: Optional[Tensor],
        kv_cache,
    ):
        # ...
    
        output: Float[Tensor, "batch num_heads seq head_dim"] = F.scaled_dot_product_attention(
            q, 
            k, 
            v, 
            attn_mask=mask, 
            dropout_p=self.attn_dropout.p if self.training else 0.0
        )
        
        output = output.transpose(1, 2)
        if self.config.attention_output_gate_enabled:
            # self.W_attn_out_gate = nn.Linear(d_model, d_model)
            gate_scores = torch.sigmoid(self.W_attn_out_gate(x))
            output = output * gate_scores.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # ...
```

[Find a complete configurable gated attention module here](https://github.com/LeonEricsson/omni/blob/a0f747bf1fcde2744334f8772bc454ffc72ad42d/omni/modules/attention.py#L229) 

Why does such a simple mechanism yield these improvements? The authors identify and analyze three primary factors.

### **Non-linearity**

In a standard multi-head attention block, the value projection ($W_V$) and the output projection ($W_O$) are two consecutive linear transformations. Their combined effect can be rewritten as a single, low-rank linear mapping. For the $k$-th head, the output for the $i$-th token is:

$$o_{i}^{k} = \left(\sum_{j=0}^{i} S_{ij}^{k} \cdot X_{j}W_{V}^{k}\right) W_{O}^{k} = \sum_{j=0}^{i} S_{ij}^{k} \cdot X_{j}(W_{V}^{k}W_{O}^{k})$$

Since the head dimension $d_k$ is typically smaller than the model dimension $d_{model}$, the product $W_V^k W_O^k$ forms a low-rank bottleneck, limiting the expressiveness of the transformation.

By inserting a non-linear function—like a gate—between these two linear layers, the model can no longer collapse them into a single mapping. This substantially increases the expressive power of the attention head. This explains why gating at positions $G_1$ (after the value-weighted sum) and $G_2$ (after the value projection) is effective, while gating at $G_5$ (after $W_O$) has a negligible effect.

### **Input-Dependent Sparsity**

While non-linearity explains part of the gain, it doesn't account for the performance difference between various gating configurations. The paper's analysis reveals that the most effective gating mechanisms are also the sparsest. The SDPA output gate ($G_1$) exhibits the lowest mean gating scores (an average of 0.116 across layers), with a distribution heavily concentrated near zero. This indicates that the gate is actively pruning a significant portion of the SDPA output.

This sparsity is also **query-dependent**. Because the gating scores for $G_1$ are computed from the current query's hidden state ($X_i$), the gate learns to filter out contextual information from the value vectors that is irrelevant to the current token. In contrast, gating at $G_2$ is less effective because its scores are dependent on the key/value hidden states ($X_j$), not the query. This query-dependent filtering may be *key*. The authors investigate further by testing an input-*independent* gate, which still provides a small boost from non-linearity but performs much worse, reinforcing that effective sparsity must be dynamic and input-dependent.

### **Reduces Attention Sinks**

A fascinating side effect of this sparse, query-dependent gating is the near-total elimination of the **"attention sink"** phenomenon that we [discussed just last week in relation to GPT-OSS](/blog/2025-08-28-rr.md). In many standard transformers, a disproportionate amount of attention is allocated to the very first token (often the `BOS` token), which acts as a kind of "garbage collector" for attention scores that have nowhere else to go. The baseline models in the study direct an average of **46.7%** of their attention to the first token. By applying the SDPA output gate, this drops to just **4.8%**.

This happens because the sparse gate effectively nullifies irrelevant attention outputs *before* they are added to the residual stream, removing the need for an attention sink. This also reduces the massive activation values often seen in hidden states, which in turn contributes to the observed training stability. With smaller activations, the model is less prone to numerical errors during mixed-precision training.

Perhaps the most practical benefit of being attention-sink-free is improved performance in **long-context extrapolation**. When extending the context length of a model post-training (e.g., using YaRN), baseline models often struggle because their reliance on the attention sink pattern doesn't adapt well to the modified RoPE frequencies.The gated models, which use query-dependent sparsity to control information flow, are far more robust to these changes and show significantly less performance degradation when extended to much longer sequences.
