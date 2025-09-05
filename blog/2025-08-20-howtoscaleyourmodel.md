---
layout: post
title: "How To Scale Your Model"
categories: []
year: 2025
type: paper
---

This is an extremely long winded, scattered, and at many times incoherent post; a result of my notetaking while reading the entire "How To Scale Your Model" series. I **highly** recommend the series, was a great read. I'd love to get more hands on experience with large scale training, would be a dream come true. 

Again, don't try and read this. This is for my own reference.
I've answered most quizzes, might be the only thing worth looking at here if your not me. 

---



### Inference

Naive sampling from a transformer. Put prompt in, get log p(next token | previous tokens). Sample from distribution, put prompt + next token in. Repeat.

This works, but we never do this in practice. Due to the causal dependency of the transformer decoder, token $n_t$ only depends on $n_{t-1}$, so, at the second step in the image above we are recomputing the same thing for all previous tokens that we already processed in step 1. The forward pass is (n²) on the FFW and O(n³) on the attention mechanism to generate n tokens, that is expensive!! 

Instead of doing the full forward pass every time, we can save some intermediate activations from each forward pass that allows us to avoid re-processing previous tokens. Specifically, since a given token only attends to the previous tokens during dot product attention, we can simply write each token's key and value projections into a new data structure called the **kv cache**. Once we've saved these key/value projections from past tokens, future tokens can simply compute their $q_i \cdot k_j$ products without performing any new FLOPs on earlier tokens. Amazing! This naturally divides inference into two separate stages

**Prefill**: This is the first step in the image above, where we have yet to process the prompt. At this step we process **all** the tokens in the prompt at the same time, saving resulting activations (specifically key-value projections) in a KV cache. We also save the logits for the last token.
**Generation**: Given a KV cache and the previous logit, we sample a new token and feed that token into the Transformer and produce a new set of logits. We also append the new KV activations to the KV cache. 

Here's a new visualization with a KV cache

By sampling with a KV cache we reduced our time complexity to generate n tokens to O(n) in the FFW and O(n²) on the attention, since we never reprocess a previous token. We will see that prefill and generate are two **very** different tasks, with the KV cache being a novel and significant source of complexity.

**What do we want to optimize?**
A part of inference that's totally new compared to training: *latency*. During training we focus on throughput, the total tokens processed per seconds, during inference we have to worry about how fast we're producing tokens, measured as both **Time to First Token** (TTFT) and the **per token latency**. This is different for different use cases:

- Chat interfaces / streaming tasks need to run cheaply at while while having a low TTFT, generating tokens fast enough to exceed human speed
- Offline batch inference for evals and data generation only care about the bulk cost of inference and is blind to the latency of individual samples
- Edge inference only needs to service one user at a time at the lowest possible latency.

Maximizing hardware utilization is still critical and helps with cost and TTFT, but unlike training it does not *necessarily* translate to better experience for individual users in all contexts. Many optimizations at the accelerator, systems and model arch level make tradeoffs between latency, throughput, context length and model quality.

#### A granular view of the Transformer
Before, when we were looking at the training perspective, we treated Transformers as a stack of MLP layers. While this is often reasonable from a FLOPs and memory standpoint, it is not sufficient  to properly model inference. The major components of the Transformer forward pass are:

1. **a bunch of linear operations**, including the MLP: W_in and W_out; the attention QKVO projections: W_Q, W_K, W_V, W_O. These all involve reading parameters and a batch of activations from HBM, doing some flops and then writing the result back to HBM.
2. **dot product attention** We need to read a batch of key-value projections and a batch of query activations from HBM, do a few inner products and some softmax operations and write back to HBM.
3. **everything else** including layer norms, activation functions, token sampling, updating kv cache and pos embeddings. These take some FLOPs but are dominated by, or fused into the above

#### Linear operations: what bottlenecks us?
Let's look at one of the linear operations, which take the form of a bf16[B, D] batch by a bf16[D, F] weight matrix. This could be either one of the big W_in/out in the MLP block or one of the smaller attention projections. To perform this matmul we need to load into HBM, perform the matmul, and store back into HBM. That means we have to move 2BD + 2DF weights into HBM, perform matmul, and then store back 2BF. Let's assume a TPU v5e, the time this takes is given by

T_comms = bytes moved / bandwidth = 2(BD+DF+BF) / W_HBM 

Then the matmul we are performing is obviously 2BDF FLOPs, and the time it takes it

T_math = computation FLOPs / accelerator FLOPs = 2BDF / C

We are compute bound if 

T_math > T_comms = computation FLOPs / accelerator FLOPs > bytes moved / bandwidth = computation FLOPs / bytes moved > accelerator FLOPs / bandwidth = intensity(algorithm) > intensity(TPU v5e)

where intensity(TPU v5e BF16) = 1.97e14 / 8.1e11 = 243

With this we get that 

2BDF / 2(BD+DF+BF)> 243

which can have different characteristics depending on the size relationbetween B,D,F. Typically F>D>>B which gives

BDF / DF(B/F + 1 + B/D) <-> B -> B > 243 = B_crit

If we quantize our weights or use lower precision FLOPs for the matrix multiplication this critical batch size can change. For instance if our weights are quantized in int8 the bytes we get

2BDF / (2BD + DF + 2BF) <-> 2BDF / DF(2B/F + 1 + 2B/D) <-> 2B -> B_crit = 243/2

or if we do our FLOPs int8 / fp8 we now load everything in int8 meaning

2BDF / BD+DF+BF <-> 2B -> 2B > HMB int8 intensity = 3.94e14 -> B > 243

so basically nothing changes if we do things in int8. We are moving 2x less data which reduces communication load, but our accelerator is 2x faster so it evens out.

We can draw some general conclusions from this; if we let $\beta$ = bits per param / bits per activation, and alpha_hbm = intensity(accelerator) = C/W_hbm, then our critical batch size is
B_crit = $\beta*\alpha$.

	Takeaway: Transforme matmuls are compute bound iff the per replica token batch size is greater than B_crit = C/W_hbm * (bits per param / bits per activation) = beta*alpha. For bf16 activationson a TPU v5e this is 240 tokens, for an h100 this is about 280 tokens. 

Remember that batch size here refers to the token batch size. During training, we'll have a very high algorithmic intensity because we reuse the same weights over a very large batch. This high intensity carries over to prefill since user prompts are typically hundreds if not thousands of tokens long. If a sequence is longer than 240 tokens and fed into a dense model we expect it to be compute-bound and all is well. Prompts shorter than this can technically be batched together to achieve higher utilization but this is typically not necessary.

	Takeaway: During prefill, all matrix multiplications are basically always compute-bound. Therefore simply maximizing hardward utilization or MFU is enough to maximize throughput per chip (cost) and latency (in the form of TTFT). Unless prompts are extremely short, batching at a per-prompt level only adds latency for a small improvements in prefill throughput.

However, when we move to the decoding/generation stage we can only do our forward passes one token at a time. Thus we can only (easily) achieve good utilization by batching multiple requests together, parallelizing over the batch dimension. Apparently, batching over concurrent requests is hard without affecting latency, for that reason it is much harder to saturate the hardware FLOPs with generation. 

	Takeaway: During generation, the total token batch size must be greater than B_crit to be compute bound on the linear/feed forward operations. Because generation is only done on one token this requires batching multiple requests, which is hard

You have to realize that handling **240 concurrent requests** means handling 240 separate KV caches. That means this is difficult to achieve in practice. In contrast, pushing more than 240 tokens through during the prefill is pretty routine. 


#### Attention!

Things get more complicated as we turn to Attention :)  Looking at pure multi head scaled dot product attention. In a single Flash Attention fusion we, ignoring
softmax, masks etc, we:

- Read Q activations of shape bf16[B, T, D] (assuming D=NH) from HBM
- Read the KV cache which is a pair of bf16[B, S, D] tensors from HBM
- Perform 2BTSD FLOPs in the QK matmul, with flash attention we dont need to write bf16[B,S,T] attention matrix mack into HBM
- Perform AV matmul taking 2BTSD FLOPs
- Write the resulting bf16[B,T,D] tensor back into HBM.

Putting this together we get 

Multihead attention intensity = FLOPs / bytes moved = 4BTSD / 2BTD + 2BSD + 2BTD = TS / T + S

During prefill S=T giving us T/2. This is great becuse it means the arithmetic intensity of attention during prefill is O(T). That means it is quite easy to be compute-bound ofr attention, as long as our sequence length is fairly large. But, during generation S>>T = 1 giving us

ST/(T+S) = S/(S+1) = 1 as S grows. 

This is bad, since we cannot do anything to improve the arithmetic intensity of attention during generation. We're doing a tiny amount of FLOPs while loading a massive KV cache. So we are basically always memory bandwidth bound during attention.

	Takeaway: During prefill, attention is typically comput bound for any reasonable sequence length (roughly > 480 on a v5e), while during generation our arithmetic intensity is roughly 1 and constant, so we are ALWAYS memory bandwidth bound.

Let's think about this. During the linear portions of the model we are compute bound because the parameters (the memory bandwidth heavy components) are reused over many batch items. However, every batch item has its own KV cache, so a bigger batch size means more kv caches. We will almost always be memory bound here unless the architecture is adjusted. 

### Theoretical estimates for LLM latency and throughput

$$\begin{align} \text{Theoretical Step Time (General)} = \underbrace{\frac{\text{Batch Size} \times \text{KV Cache Size}}{\text{Total Memory Bandwidth}}}_{\text{Attention (always bandwidth-bound)}} + \underbrace{\max\left(\frac{2 \times \text{Batch Size} \times \text{Parameter Count}}{\text{Total FLOPs/s}}, \frac{\text{Parameter Size}}{\text{Total Memory Bandwidth}}\right)}_{\text{MLP (can be compute-bound)}} \end{align}$$

Throughput / latency pareto charts from PaLM. We can trade throughput for latency up to a certain point. As the batch size goes beyond the 240 mark, our MLP FLOPs begin to dominate over
communication time, and as such the throughput then starts to depend on the batch size, meaning tha throughput is flat as we increase batch size beyond that point. Before that, communication time
dominates, which depends only on parameter size and bandwidth.  Once MLP becomes compute bound the throughput is given by

	Throughput = Batch Size / (Attention Time + MLP Time)
	           = Batch Size / (Batch Size × KV_factor + Batch Size × MLP_factor)
	           = 1 / (KV_factor + MLP_factor)
           
which scales linearly with batch size.

	Takeaway: If we care about generation throughput, use the largest per-chip batch size possible. Any per-chip batch size above the arithmetic intensity (B_crit which is typically 120 or
	240 depending on quantization) will maximize throughput. You may need to increase topology to achieve this. Smaller batch sizes will allow you to improve latency at the cost of throughput.

#### What about memory?
We've looked at bandwidth and FLOPs of attention and linear operations during inference, but not the memory. Memory looks quite different during inference thanks to the kv cache. During training, we have parameters, activations, optimizer states. Where activations typically dominate the memory requirements. During inference, many of the things we have to store during training disappear. We don't have a optimizer, and we don't perform a backward pass so we dont need to save activations. It's actually just the parameters, with the addition of the kv cache. For the coming section, let's look at a real model to demonstrate how different things are in inference.

```
LLAMA 2 13B
L = 40
D = 5120
F = 2.7*D
N = 40
K = 40
H = 128
V = 32000
```

As we said, during inference our parameters require memory. Counting these we have, per layer:

FFW: 3DF 
Attention: DNH + DKH + DKH + NHD = {N=K} = 4DNH
Vocab = 2DV

Adding these up gives 13e9 parameters, as expected. As we saw in the last section, storing parameters in bf16, with optimizer state in float32 may use around 100GB (2 bytes for params, 4 for m_t and 4 for v_t). This pales in comparison to the gradient checkpoints (activations) which can take several TBs.

During inference we only store one copy of params, in something like bf16 using at most 26GB. But we can often do even better with quantization. Activations are negligable. There are no optimizer states.
The main difference is the **kv cache**. The total size of the kv cache for T tokens is

KV cache size = 2 * bytes per float * L * T * K * H

where H is the dimension of each head, K the number of KV heads, L layers and 2 comes from storing both K and V. This can get big fast. For our model at hand, a 8192 sequence at bf16

(2 * 2 * L * T * K * H) / 1e9 = 6.7GB

**At just a batch size 4 we've exceeded the memory usage of our parameters.** 

#### Modelling throughput and latency for Llama 2 13B
What happens when we want to perform inference at different batch sizes on 8x TPU v5e

| Batch Size                        | 1      | 8      | 16     | 32     | 64     | 240    |
| --------------------------------- | ------ | ------ | ------ | ------ | ------ | ------ |
| KV Cache Memory (GiB)             | 6.7    | 53.6   | 107.2  | 214.4  | 428.8  | 1608   |
| Total Memory (GiB)                | 32.7   | 79.6   | 133.2  | 240.4  | 454.8  | 1634   |
| Theoretical Step Time (ms)        | 4.98   | 12.13  | 20.30  | 36.65  | 69.33  | 249.09 |
| Theoretical Throughput (tokens/s) | 200.61 | 659.30 | 787.99 | 873.21 | 923.13 | 963.53 |
We note that the KV Cache dominates our memory footprint, ammortizing the parameter cost. 8x TPU v5e give us 128GB of HBM, 6.5 TiB/s of HBM bandwidth and 1600TF/s of compute. Increasing the batch size increases our throughput, as we expect, but at diminishing returns. We will OOM at batch sizes > 16. If we keep the number of params the same, but are able to magically make our KV cache 5x smaller:

| Batch Size                        | 1      | 8        | 16       | 32       | 64       | 240      |
| --------------------------------- | ------ | -------- | -------- | -------- | -------- | -------- |
| KV Cache Memory (GiB)             | 1.34   | 10.72    | 21.44    | 42.88    | 85.76    | 321.6    |
| Total Memory (GiB)                | 27.34  | 36.72    | 47.44    | 68.88    | 111.76   | 347.6    |
| Theoretical Step Time (ms)        | 4.17   | 5.60     | 7.23     | 10.50    | 17.04    | 52.99    |
| Theoretical Throughput (tokens/s) | 239.94 | 1,429.19 | 2,212.48 | 3,047.62 | 3,756.62 | 4,529.34 |
|                                   |        |          |          |          |          |          |
Now, we will OOM at a batch size of > 64. We still see diminishing returns but throughput scales beter up to 240. 

	Takeaway: The size of the KV cache has a lot of bearing over the ultimate inference performance of the model. At longer sequence lengths the attention time dominates MLP time, which means that reducing the KV cache size by a factor 1/X will roughly reduce the step time by the same factor 1/X (and increase throughput by X).


### Tricks for improving generation throughput and latency
Many techniques have been developed targeting the KV cache specifically. 

**MQA, GQA, MLA**

**Mixing in local attention**: Local attention caps the context to a small-moderate size max length. At training time and prefil time, this involves masking the attention matrix to a diagonal strip instead of triangle.

**Sharing KV cache across layers**: The model can learn to share the same KV cache across layers in some pattern. While this benefits KV cache size and provides benefits to increasing batch size, caching, shared KV caches may need to read from HBM multiple times so it does not necessarily improve step time.

**Quantization** Inference is less sensitive to the precision of parameters and the KVs. By quantizing the parameters and KV cache (eg to int8, int 4, fp8 etc) we can save on memory bandwidth on both, decrease the batch size required to reach the compute roofline and save memory to run at bigger batch sizes. Quantization has the added advantage that even if the model was not trained with quantization it can be applied post training.

**Using ragged HBM reads and Paged Attention** We allocate 8k of context for each KV cache but it is often not necessary to read the entire KV cache from memory - requests have a wide range of length distributions and dont use the max context of the model. 

Paged Attention is a refinement upon this that stores KV caches in OS-style page tables and mostly avoids padding the KV caches altogether. This adds a lot of complexity but means every batch only uses as much memory as it needs. Instead of allocating a standard 8k to every batch request we instead only use the necessary amount for each request. 

	Big Picture: All told, these KV cache optimizations can reduce KV cache sizes by over an order of magnitude compared to a standard MHA Transformer. This can lead to an order-of-magnitude improvement in the overall cost of the Transformer.

#### Distributing Inference Over Multiple Accelerators

We've mostly handwaved how we are scaling beyond a single chip. Let's look at this now, prefill and generation separately.

**Prefill**
The roofline calculations are almost identical to training and almost all the same techniques apply - model (megatron) parallelism, sequence sharding (for sufficient long context), pipelining, even FSDP
are all viable! You just have to keep the KVs kicking around so you can do generation later. As in training, increasing the number of chips gives us access to more FLOPs but adds communication overhead. 
General rule for sharding prefill: Assuming we're doing prefill on a single sequence ( no batch dim):
1. Model sharding: we typically do some amount of model parallelism first, up to the point we become ICI-bound. From section 5, this is around F/2550 for 1 axis. 
2. Sequence parallelism: Beyond this we do sequence parallelism (like data parallelism but sharding across the sequence dimension). While SP introduces some extra communication in attention, it is typically fairly small at longer contexts. As with training we can overlap comms and math.

	Takeaway: during prefill, almost any sharding that can work during training can work fine. Do model parallelism up to ICI bound, then sequence parallelism

**Generation**
is a different beast. For one thing, it is harder to get a large batch size since we need to batch many requests together. Latency targets are lower. Together, this means we are typically more memory bound
and more sensitive to communication overhead.

- **FSDP is impossible**. We are memory bound in loading our parameters and KV caches from HBM to MXU, we do not want to move them via ICI which is orders of magnitude slower than HBM. If anything we want to **move activations rather than weights**. Activations are considerably smaller. 
- **There is no reason to do data parallelism**. Pure data parallelism is unhelpful, we are already memory bound on a single chip and DP replicates parameters, which doesn't make parameter loading faster. You're better off spinning up multiple copies of the model instead.  
- **No sequence = no sequence sharding**

This mostly leaves us with model sharding for dense model generation. 

**Note on ICI bounds for generation**. During training we want to be compute bound, hence we try and identify at what point our ICI comms take longer than our FLOPS. However, during generation, if we're memory bound (HBM to MXU) by parameter loading, we can increase model sharding beyond the aforementioned point and improve latency at a minimal throughput cost. More model sharding gives us more HBM to load our weights over, and our FLOPs dont matter (in the sense that FLOPs time isnt the bottleneck, so the thing we need to worry about is ICI time exceeding parameter loading time). 

T_HBM_comms = 2DF / YW_hbm
T_ICI_comms = 2BD / W_ICI

T_ICI_comms > T_HBM_comms <-> 2BD/W_ICI > 2DF / YW_hbm <-> W_hbm / W_ICI > F/BY <-> Y > F/B * beta

Beta is the ratio between hbm and ici speed, which is usually around 8. That means, for the llama model above we have Y = 54 without a meaningful hit to throughput. This assumes we can fully shard our KV caches 54 ways which is difficult.

	Takeaway: our only option during generation are variants of model parallelism. We aim to move activations instead of KV caches or parameters because we are memory bound and we want to limit data transfer over ICI. When our batch size is large, we do MP up to the FLOPs-ICI bound (F/alpha). When our batch size is smaller we can improve latency by model sharding more. When we want to model shard more ways than we have KV heads we can shard our KVs along the batch dimension as well.

#### Sharding the KV cache

We almost always prefer to avoid replicating the cache, since it is the primary source of attention latency. To do this, we megatron shard across the head dimension, which limits us to K way sharding, so for models with a small number of heads we shard the head dimension as much as possible and then shard the batch dimension. Given a KV cache [2, B, S, K, H] we shard it as [2, Bz, S, Ky, H]. This means the KV cache is completely distributed.

X[B, D] = (existing activations, unsharded from previous layer)
K[Bz, S, Ky, H], V[Bz, S, Ky, H] = ... (existing KV cache, batch sharded)

Q[B, Nyz, H] = X[B, D]  x W_Q[D, Nyz, H]
Q[Bz, Ny, H] = AllToAll_z->b(Q[B, Nyz, H])
Q[Bz, Ky, M, H] = Q[Bz, Ny, H] (split N -> K, M)
O[Bz, S, Ky, M] =  Q[Bz, Ky, M, H] x K[Bz, S, Ky, H] 
O[Bz, S, Ky, M] = softmax(O[Bz, S, Ky, M])
O[Bz, Ky, M, H] = O[Bz, S, Ky, M] x V[Bz, S, Ky, H]
O[B, Ky, Mz, H] = AllToAll_z->M (O[Bz, Ky, M, H])
O[B, Nyz, H] = Reshape(O[B, Ky, Mz, H])
X[B, D]{Uyz} = O[B, Nyz, H] x W_O[Nyz, H, D]
X[B, D] = AllReduce(X[B,D] {U_yz})

This is kind of complicated, byt we can see that sharding over the batch dimension like this requires us to perform2 AllToAll collectives, one to shift the Q activations to the batch sharding so we can compute attention with batch sharding, and one to shift sharded attention output back to pure model sharded.  The new comms are modestly expensive since they operate on our small activations, while in return we save a huge amount of memory bandwidth loading the KVs. 

### Designing an effective inference engine

So far we've looked at how to optimize and shard individual prefill and generate operations in isolation. How do we combine these? 
The simplest method is simply run a batch of prefill, then a batch of generations
This is easy to imlpement and is the inference setup in most codebases, but it has multiple drawbacks:
- Latency is terribl. We couple the prefill and generate batch size. Time to first token is terrible at big prefill batch sizes - you need to finish all prefills before
  any user can see any tokens. Generate throughput is terrible at small batch sizes
- We block shorter generations on longer ones. Many sequences will finish before others, leaving empty batch slots during generation. 
- Prefills are padded to the longest sequence and we waste a lot of compute. 


A slightly better approach involves performing prefill at batch size 1 (where it is compute bound but has reasonable latency), but batch multiple requests during generation. This will avoid wasted TTFT from batched prefill while keeping generation throughput high. We call this interleaved configuration since we interleave prefill and generation steps. This is very powerful for bulk generation applications like evaluations where throughput is the main goal. We want to batch generation to improve throughput, and because prefill is already compute bound at batch size 1, this combination achieves this well. However, if we are serving a user, this configuration can lead to jittery and slow response on average. Other user prefills are placed on the critical path of the overall latency of a request.

To get around this, we separate decode and prefill.

#### Serving LLama

**Question:** Now let’s dig into the question of sharding. Let’s say we wanted to serve in bfloat16 on a TPU v5e 4x8. What sharding would we use for our model on a TPU v5e 4x8 during generation? Can we avoid being communication bound?

The only parallelism we can apply is model parallelism. TP becomes ICI bound when Y > n_axes *  F / 2200 = 26. That means, with a 4x8 configuration we can not apply TP. The most we can do is 4x4, and even that might be pushing it considering we can not always perfectly overlap comms and math.

But, remember that during generation, we are in the memory bandwidth bound regime due to parameter loading, which means we can increase model sharding beyond the traditional point at a minimal throughput cost. More model sharding means more HBM to load our weights over and our FLOPs done "matter" in the sense that FLOP time isnt bottlenecking us. All we need to worry about is ICI time exceeding parameter loading time.

T_comms_hbm = 2DF/YW_hbm
T_comms_ici = 2BD/W_ICI

T_comms_ici > T_comms_hbm <-> 2BD/W_ICI > 2DF/YW_hbm  <-> F/BY < q <-> Y > F/Bq =  3185/B

We know that we have 32 GPUs which means we can our batch size can at most be 99. As long as our batch size is less than this we will be HBM bound on 32 GPUs. We can sanity check this further by looking at the raw values for a 4x8 and a 64 batch size

T_comms_hbm = 2DF/YW_hbm = 0.018ms
T_comms_ici = 2BD/W_ICI = 0.011ms
T_math = 2BDF/YC = 0.0047ms

	Takeaway: the maximum amount of useful model parallelism depends on d_ff and the number of axes over which you're sharding the model. This value typically ranges between 8 and 32 depending on model size, the larger the model, the larger you can parallelize before being ICI bound. You can scale beyond this point this limit to improve latency at some throughput cost.

**Prefill**
We've mostly ignored prefill because it is much simpler to deal with. Let's put a couple of concepts together and think about the end-to-end picture.

**Question:** Assume we achieve a 40% FLOPs utilization during prefill. How long will a prefill of length 8192 take on 16 TPU v5e chips?

A 40% FLOPs utilization means we are achieving 16 * 1.97e14 * 0.4 = 1.2608e15 FLOPs. How many FLOPs are required for a 8192 prefill? At 8192 tokens we are solidly in the compute bound regime. The forward pass uses 2 * num params * num tokens FLOPS. Which means this takes 0.9 seconds. 

**Question:** Assume we have a median prefill length of 8192 tokens and a median decode length of 4096 tokens. Say we have a generate batch size of 32. On average how many sequences finish decoding per step? On average how many tokens are evicted from our KV cache each step?

We decode one token per step, and each sequence needs to decode 4096 tokens. At a batch size of 32 that means 32/4096 sequences finishing every step.

Our KV cache length is 8192 + 4096 (assuming a fixed size). This KV cache is dropped when we finish a sequence. So that means we are dropping 8192 + 4096 * 32/4096 = 96 tokens every step.

**Question:** Assume we do disaggregated serving with a median prefill length of 8192 and a median decode length of 512. Assume the prefill and generate latencies calculated above in bfloat16. What ratio of prefill:generate servers will you need to keep both fully saturated.

Prefill latency for 8192 was 0.91 seconds
Generation latency for decode length 512 is 0.019s at batch size 32 (was 43 but lets say 32)

Let P be the number of prefill servers and G the number of generation servers. We will feed sequences into generation at a rate of P / prefill latency and consume them at a rate of B * G / (latency * decode steps). This gives

P/0.91 = 32G/(512 * 0.019) <-> P =  3G 

so we need 3 times more generation servers than prefill servers.

As we've established before, during **generation**, the time is dominated by parameter loading at small batch sizes. As we cross a batch size of ~120 (this it int8) we become compute bound in the MLPs and the FLOPs start
to dominate our time share. However, as we increase context, the only parameter that increases is the KV comms, we have to move a lot more bytes around, which means at increasing context lengths, in just case at just 4096, the KV comms are larger than the FLOPs. Naturally the KV cache size grows with batch size so at significant batch sizes the KV cache is what dominates the total time share. At context lengths of 16384 we cant even increase our batch size enough to reach the MLP compute bound regime anymore, as the context grows the total memory usage of the KV cache means our maximum batch size shrinks.  

	Takeaway: for LLaMA 3-70B, we are strongly KV cache memory bandwidth-bound (and HBM-bound) in almost all of these configurations, highlighting just how important reducing KV cache size is for generation throughput. Also note just how dramatic the latency/throughput tradeoff remains here.

**Question 1:** How many FLOPs does each forward pass for LLaMA 3-405B use per-token? Assuming we’re FLOPs bound, what is a lower bound on a single forward pass on N chips on TPU v5e? What if we’re comms bound? _Ignore the fact that the model does not fit on a single chip._

The FLOPs in a forward pass are 2 * num params per token  which gives 810e9 FLOPs. Assuming we are FLOPs bound,, that means our lower bound is just the time it takes for our N cards to perform the necessary FLOPs of the forward pass. The TPUv5e has 1.97e14 FLOPs so the answer is:

810e9 / (N * 1.97e14)

If we are comms bound, that means the lower bound is given by the time it takes to move our parameters into MXU. Not sure if they are assuming N cards here, but in the single chip case the lower bound is

2 * 405e9 / 8.1e11

because the TPUv5e HBM BW is 8.1e11.

If we are comms bound as in ICI bound, then we assume the model is sharded over N chips and we get

2 * 405e9 / (N * W_ICI) = 2 * 405e9 / (N * 9e10)

**Question 2:** Assume we want to serve LLaMA 3-8B with BS240 using int8 weights and int8 KV caches. How many bytes are used by (a) model parameters 
b) KV caches and (c) peak working activations (roughly)? What’s the smallest topology we can run this on?

Bytes
a) 8GB
b) KV cache is [2, bytes per param, L, K, N] per token where L the layers, K the number of KV heads and N the head dimension. The config for LLama 3 8B gives us
2 * 1 * 56 * 8 * 128 = 114KB per token. With batch size 240 we know that the total kv cache bytes is 114e3 * 240 * S where S is the context length.
c) ignoring actuvations because they are roughly negligible.

To determine the smallest topology we can run this on, lets assume our context length is 2048, that means the KV cache requires 56GB. We therefore require 64GB, meaning a 4x2 is sufficient (8 * 16GB). If we 
want to increase to 4096 context length we need 120 GB which will barely fit on 4x2 given the overhead.

**Question 3:** How would you serve LLaMA 3-405B on TPU v5e? Assume int8 weights and bfloat16 FLOPs. 
Let’s say we have a firm limit of 15ms / token, what’s the highest throughput configuration we could achieve? What is the theoretical minimum step time?

Let's see. Under this configuration our highest throughput is when we become compute bound by our MLPs at B_crit = 120.

The question is what throughput are we achieving at 15ms / token. 

We have the general step time formula as 
$$\begin{align} \text{Theoretical Step Time (General)} = \underbrace{\frac{\text{Batch Size} \times \text{KV Cache Size}}{\text{Total Memory Bandwidth}}}_{\text{Attention (always bandwidth-bound)}} + \underbrace{\max\left(\frac{2 \times \text{Batch Size} \times \text{Parameter Count}}{\text{Total FLOPs/s}}, \frac{\text{Parameter Size}}{\text{Total Memory Bandwidth}}\right)}_{\text{MLP (can be compute-bound)}} <= 15ms\end{align}$$
For int8 parameters and bf16 FLOPs the MLP turns compute bound when per-chip batch size is > B_crit = 120.

Let's perform some sanity checks, starting with seeing if the MLP can even be compute bound with the latency limit.

When the MLP becomes compute bound we have 

2BP / (NC) < 15ms <-> B/N < 3.6 token/ second / chip 

which is too large a latency meaning we will not be able to reach the B=120 compute roofline under the 15ms cap, the MLP will be comms bound not compute bound. Looking at the parameter loading

bytes / (N * W_hbm) = 0.5/N <0.015 <-> N > 33.33

means we need more than 34 chips to hide the parameter loading in the 15ms limit.  A 4x8 topology is too small, we need to move to an 8x8 with 64 chips. At this size, the MLP step time is

0.5/N = 7.8ms

which leaves 7.2ms for the attention computation. The KV cache is [2, bytes per param, L, K, H] per token, which for our given 405 config is 2 * 1 * 126 * 8 * 128 = 258KB per token. 

Batch size * sequence length * kv cache per token / (N * W_HBM) < 7.2ms <-> BS < 1446697

## GPUs

## Networking

#### Node level

Performing an AllGather of bf[Dx, F] in a H100 node. Let's assume N=X, we have to communicate 2DF/N bytes from each GPU, where each GPU has a unidirectional bandwidth of 450GB/s, which gives 2DF/(NW_uni). This is performed N-1 times in a ring style AllGather implementation. Giving a total communication time of

(N-1)* 2DF / (N * W_uni)

#### Beyond the node level

TPU v5p has about 90GB/s egress bandwidth per link, 540GB/s along all axes of the 3D torus. Within the H100 node, we have 450GB/s from each GPU, while beyond the node, this drops to 400GB/s node-to-node. 

The newly released GB200 NVL72 SuperPods drastically change our rooflines. This is the first time we are moving beyond 8 GPUs in our NVLink domain, which is now increased to 72! These NVLinks now do full 900GB/s of GPU to GPU bandwidth. 

**Question 1 [Fat tree topology]:** Using the DGX H100 diagram above, calculate the bisection bandwidth of the entire 1024 GPU pod at the node level. Show that the bandwidth of each link is chosen to ensure full bisection bandwidth. _Hint: make sure to calculate both the link bandwidth and switch bandwidth._

Any even partition of this will include 2 SUs. Let's first look at the node->leaf connections. There are 8 leafs, each leaf is a 64 port NDR IB switch with 50GB/s per port, but we can only use 32 ports to ingress. That means the total switch bandwidth of the SU is 32 * 50 * 8 = 12.8TB/s. That means per node our bandwidth is 12.8/32 = 400GB/s.  Let's now look at the link bandwidth, each node is connected through 8 links to the switches, this gives 3.2TB/s of total egress from the node. Per GPU this means 400GB/s. 

At the spine level each leaf is connected to the 16 spines via 2x400GB/s. There are 8 leafs in the SU. That means the SU is connected to the spline with 8 * 16 * 2 * 400 / 8 = 12.8TB/s per leaf. That means, per node we get  400GB/s to the spline.  

The splines are 16 switches with 64 ports. The total switch bandwidth is 51.2TB/s which, at 128 nodes is 400GB/s per node. 

Thus if we bisect our nodes in any way, we will have 400GB/s per GPU between them. Every component has exactly the requisite bandwidth to ensure the fat tree.

**Question 2 [Scaling to a larger DGX pod]:** Say we wanted to train on 2048 GPUs instead of 1024. What would be the simplest/best way to modify the above DGX topology to handle this? What about 4096? _Hint: there’s no single correct answer, but try to keep costs down. Keep link capacity in mind. [This](https://docs.nvidia.com/dgx-superpod-reference-architecture-dgx-h100.pdf) documentation may be helpful._

We can't increase the number of GPUs in a SU because our IB switches are at their max capacity of 32 ingress wires. We would have to double the amount of scalable units to 8. Our spine switches still have 32 ports available. This means we would have to increase to 32 spines. 

Let's do the math. We have 8 SU, each with 8 leafs, and 32 GPUs.

The leaf is connected to 32 spines, with 1 * 50 GB/s, that means we have 400GB/s of per node link BW. The switch bandwidth is 64 * 32 * 50 = 102.4 TB/s which is 400GB/s per node given that we have 256 nodes. This would work. We can use the same setup as before within SUs and just double the SUs and splines. The only difference is that we connect our leaf-splines with 1x NDR instead of 2. 

For 4096 GPUs we run out of ports, so we would need to add another level of indirection, that is to say another level of hierarchy. NVIDIA calls this level core switches. One easily notes how much more complexity is added in the system with the tree like structure compare to TPU pods. 

## How do collectives work on GPUs?

GPUs can perform all the same collectives as TPUs: AllReduce, AllGather, ReduceScatters, and  AllToAlls. Unlike TPUs however, the way these work depends on whether they are performed at the node level (over NVLink) or above (over InfiniBand). The collectives are imlpemented by NVIDIA in the NVSHMEM and NCCL libraries. NCCL uses a variety of implementations depending on latency requirements/topology, but in this section we discuss a theoretically optimal model over a switched tree fabric.

#### Intra-node collectives
**AllGather or ReduceScatter**: For an AllGather or a ReduceScatter at the node level, we can perform them around a ring just like a TPU, using the full GPU-to-GPU BW at each hop.  We can imagine this as each GPU sending B bytes over the network using the egress or unidirectional bandwidth. The cost of this hop is T_hop = bytes/(N * GPU egress bandwidth), where bytes is the total bytes across all devices. The overall cost is therefore

T = (N-1) * bytes / (N * W_uni) -> bytes/w_uni = V/W_uni

Note that this is exactly the same as on a TPU. For an AllReduce we combine AllGather + ReduceScatter as usual for twice the cost. 

In cases where the array is very small, we can do a tree reduction where we allreduce within pairs of 2, then 4 then 8 for a total of log(N) hops instead of N-1. Obviously the total cost is still the same.

	Takeaway: the cost to AllGather or ReduceScatter an array of B bytes within a single node is T_comms = B * (8 - 1) / (8*W_uni) = B/W_uni. This is theoretically around B/450e9 on a H100 and B/900e9 on a B200. An AllReduce is 2x this cost.

**Pop Quiz**: T_comms = 2BF * 7 / (8 * 450e9) = 65us 

**AllToAll** As opposed to TPUs, GPUs within a node have all-to-all connectivity, making AllToAlls simple. Each GPU sends directly to destination node. Within a node, each GPU has B/N bytes and sends (B/N²) bytes to N-1 targets for a total of

T_comms = B * (N - 1) / (W * N²)  ≈ B/(WN)

Compare this to a TPU where the cost is B/4W. Within a single node we get a 2x theoretical speedup in time B/8W

**Pop Quiz** Under non sparse conditions this will take

T_comms = B/WN.

If we know that 4 out of 8 entries will be non zero, we get 

T_comms = B / (2WN)

	Takeaway: A AllToAll collective performed on an array of B bytes, on a single node, is T = B/(8W_uni), meaning 1/8th the cost of an AllGather. In comparison, it is B/4W on TPUs. For a ragged tensor (top-k), this is decreased further to B*k/(64W_uni). 

#### Cross-node collectives

As repetition, the cost to AllGather or Reduce scatter at the intra-node level of NVIDIA GPUS is given by the following. At the intra node level we have N gpus, B bytes. Each device wants to communicate B/N bytes. Due to the node setup we have direct connectivity between ALL devices in the node. That means each device wants to egress B/N bytes, to N-1 GPUs, and it can do that at the available GPU agress bandwidth. That means the cost of each hop is $T_{hop} = B/(N * W_{\text{gpu egress}})$  so the overall cost is
$$
T_{\text{intra-node AllGather or ReduceScatter}} = \frac{B * (N-1)}{(N * W_{\text{gpu egress}})} \approx \frac{B}{W_\text{gpu egress}}
$$

which you will note is the same as for TPUs. Similarly, the cost for an AllReduce is the combination of RS+AG, at twice the cost

$$
T_{\text{AllReduce}}  \approx \frac{2*B}{W_\text{gpu egress}}
$$

**AllGather and Reduce Scatter**

Now, on-to **cross-node collectives**. When doing a reduction over a tree you can think of reducing bottom up, first within the node, then at the leaf level and then at the spine level. This has the nice effect that for an AllReduce, we communicate less data overall because we will reduce at the node level and we only have to egress $B$ bytes up to the leaf instead of $B*N$. Because we have full bisection bandwidth (the smallest bandwidth between any even partition of the network is equal to our full bandwidth) the cost of an AllGather or ReduceScatter is roughly the buffer size in bytes divided by the node egress bandwidth:

$$
T_{\text{cross-node AllGather or ReduceScatter}} \approx \frac{bytes}{W_\text{node egress}} = \frac{bytes}{400e9} 
$$
You can imagine this as performing a ring reduction over every node in the cluster. Now you may be wondering, do we not have to perform the intranode reduction first, before we can do the cross-node reduction? Like often is the case, these two collectives are overlapped, and the intra node reduction will (almost) never be the bottleneck so we don't need to calculate it. But, the general cost is:

$$
T_{\text{total}} = \text{max}(T_\text{comms at node}, T_\text{comms in scale-out network}) = \text{max}[\frac{B}{W_\text{gpu egress}},\frac{B}{W_\text{node egress}}]
$$

**Precise calculation**

Let's be even more precise in this calculation. As we've established, we're effectively doing a ring reduction at each layer in the tree (network) which we can mostly overlap. That means, the cost, is whichever reduction takes the longest. A general way to write this is 

$$
T_{\text{AG or RS}} = B * \text{max}_\text{depth i}[\frac{D_i - 1}{D * W_\text{egress i}}]
$$

where $D_i$ is the degree at depth $i$, that is the number of children at depth $i$. To determine which level of the tree determines our time / BW, we just have to solve the max() part of the formula.

Node: There are 8 GPUs with egress BW of 450GB/s, this will take 7 / (8 * 450e9) = 0.0019us
Leaf: There are 32 nodes in an SU with egress BW of 400GB/s. This gives 31/(32 * 400e9) = 0.002us 
Spine: There are 4 SUs in total with egress BW of 12.8TB/s. This gives 4/(3 * 12.8e12) = 0.05ps

As we can see, the bottleneck is at the leaf level.

---

**Other collectives**

AllReduces are still 2x the above cost unless SHARP is enabled. 

AllToAlls change a bit in the cross-node because they are not hierarchical in the way AllReduces are. If we want to send data from every GPU to every other GPU we can't take advantage of the full bisection BW at the node level. That means if we have an N-way AllToAll that spans M = N/8 nodes, each node holds B/M bytes, it keeps 1/M and sends the rest to the other nodes (M-1). Giving

$$
T_{\text{cross-node AllToAll}} = \frac{\frac{B}{M} * (M - 1)}{W_\text{node egress}}  = \frac{B * (M - 1)}{M^2 * W_\text{node egress}} \approx \frac{B}{M * W_\text{node egress}}
$$

That means, when moving from a single node to two nodes, our AllToAll collectives go from $B / (8 * 450e9)$ to $B/(2 * 400e9)$.  A general formulation of this is:

$$
T_{\text{AllToAll}} =
\begin{cases}
\displaystyle \frac{B}{N * W_{\text{gpu egress}}}, & N \leq 8, \\[1.2em]
\displaystyle \frac{B}{W_{\text{node egress}} \cdot \tfrac{N}{8}}, & N > 8.
\end{cases}
$$

which for our full fat tree is

$$
T_{\text{AllToAll}} =
\begin{cases}
\displaystyle \frac{B}{N * 450e9}, & N \leq 8, \\[1.2em]
\displaystyle \frac{B}{N * 50e9}, & N > 8.
\end{cases}
$$

	Takeaway: beyond the node level, the cost of an AllGather or ReduceScatter on B bytes is roughly B/W_node egress, which is B/400e9 on a H100 DGX SuperPod.

**Reductions when array is sharded over a separate axis**

In TPU-land, performing reductions such as

$$
\text{AllReduce}_X (A[I_Y,J](U_X))
$$

where we reduce over an array that has a dimension sharded over a separate axis **reduced the cost by a factor 1/Y**.  This makes sense because we are moving 1/Y less data in each hop. Unfortunately, in GPU-land, this is not as straight forward. On GPUs, the cost depends on which axis is the "inner" one (intra-node vs inter-node) and whether each shard spans more than a single node. Going back to the general formulation 

$$
T_{\text{total}} = \text{max}(T_\text{comms at node}, T_\text{comms in scale-out network})
$$
First, look at the intra node setting, where N-1 is replaced with N for simplicity. 
$$
T_{\text{intra-node}} = \frac{B * D}{\min(Y, D) * W_{\text{gpu egress}}}
$$
Where D is the degree of the node (8). Then the scale out

$$
T_{\text{scale-out network}} = \frac{B * N}{Y * W_{\text{node egress}}}
$$

**Quiz 4: Collectives**

**Question 1 [SU AllGather]:** Consider only a single SU with M nodes and N GPUs per node. Precisely how many bytes are ingressed and egressed by the node level switch during an AllGather? What about the top-level switch?

Let's work through the components of the reduction. 

Each GPU holds B/NM bytes of data. Within each node, each GPU sends B/NM to the switch, for a total ingress of BN/NM = B/M bytes ingressed. 

The switch egresses B/M bytes to the spine switch. 

The spine switch ingresses B * M / M bytes. At this point, the spine switch holds the entire B bytes.

Now we need to send the data back down in the tree. Every node already holds B/M of the data so each node only needs what its missing: B - B/M = B(M-1)/M. That
means the spine switch will egress a total of M * B(M-1)/M  = B(M-1) bytes. Each node ingresses the B(M-1)/M.

Now, the last step is to egress downwards to each GPU. Remember, our GPUs already hold B/NM of the total data which means each GPU needs B - B/NM. Distribute that to all N GPUs in the node and the **per node egress** is
N(B - B/NM) = NB - B/M.

Lets now look at the totals:

GPU
Egress: B/NM
Ingress: B - B/NM

Node switch
Egress: B/M + NB - B/M = BN
Ingress: B/M + B - B/M = B

Spine switch
Egress: B(M-1)
Ingress: B
$T_\text{AllGather} = B(M-1) / (M * W_{node})$

**Question 2 [Single-node SHARP AR]:** Consider a single node with N GPUs per node. Precisely how many bytes are ingressed and egressed by the switch during an AllReduce using SHARP (in-network reductions)?

Each GPU sends B(N-1)/N bytes to the node switch for a total of N *  B(N-1)/N = B(N-1) bytes. Normally, at this point we would want to communicate the rest of the missing B bytes to each GPU such that they can perform the reduction B - B/N. In total the switch egress would be N(B-B/N) = BN - B. But, with SHARP we can perform partial reduction at the switch level meaning we only have to communicate the resulting B/N bytes to every GPU for a total of N * B/N = B bytes.  Then, we do partial sum of residuals locally on the GPU and send this back to the switch, N * B/N = B bytes ingressed. We then capture all the shard and multicast them, sending B(N-1)/N to N destinations for a total of B(N-1) /N * N = B(N-1) egressed.

Therefore the totals are

Node
Ingress: B(N-1) + B = BN bytes
Egress: B + B(N-1) = BN bytes

This supports the overall throughput being exactly B/W_egress

**Question 3 [Cross-node SHARP AR]:** Consider an array bf16[DX, FY] sharded over a single node of N GPUs. How long does AllReduce(bf16[D, FY] { UX }) take? You can assume we do in-network reductions. Explain how this differs if we have more than a single node?

We can try to modify the previous answer assuming sharding XY. Each GPU sends B(X-1)/XY bytes, then send back B/XY to each GPU, then send the same amountback to the switch, then send B(X-1)/XY back to each GPU. The total is BN/Y ingress and egress which means the total time is

BN/(Y * N * W_link) = N 2DF / (Y N W_link) = 2DF/(YW_link)

**Question 5 [2-way AllGather cost]:** Calculate the precide cost of an AllGather of B bytes over exactly 2 nodes. _Make sure to calculate the precise cost and not the approximation, and consider both the intra-node and cross-node cost._

$$
T_{\text{AG or RS}} = B * \text{max}_\text{depth i}[\frac{D_i - 1}{D * W_\text{egress i}}]
$$

First lets look at the intra node cost:

$$
T_{\text{AG or RS}} = B * [\frac{8 - 1}{8 * W_\text{GPU egress}}] = B/514e9
$$

now the cross node

$$
T_{\text{AG or RS}} = B * [\frac{2 - 1}{2 * W_\text{node egress}}] = B/800e9
$$
which means that we are bottlenecked by the intra node reduction not the leaf level. This motivates 2-way DP. 

### Rooflines for LLM Scaling on GPUs

The idea of this chapter is to compare $T_\text{math}$ and $T_\text{comms}$ for different parallelism strategies and understand at what point $T_\text{comms} > T_\text{math}$. This tells us when a certain parallelism strategy has run its course, and we've become bottlenecked by our communication collectives as opposed to our compute FLOPs. As before, we consider only the MLP block with operations

MLP = x[B, D] * W_in[D, F] * W_out[F, D]

where B is the global batch size **in tokens** (i.e B = batch size * sequence length).

| Node Type   | GPUs per node | GPU egress bandwidth | Node egress bandwidth |
| ----------- | ------------- | -------------------- | --------------------- |
| H100        | 8             | 450e9                | 400e9                 |
| B200        | 8             | 900e9                | 400e9                 |
| GB200 NVL72 | 72            | 900e9                | 3600e9                |

Both GPU and node egress bandwidth determine rooflines for our LLMs. We use $W_\text{collective}$ to describe either the GPU or node bandwidths depending on whether we are operating within or above the node level.

#### Data Parallelism

The cost of pure DP or FSDP without network reductions, per layer, in the backward pass with an axis size of X. In the backward pass we have four matmuls, each of which requires 2BDF FLOPs. Thus for a single layer:

T_math = 4 * 2BDF / XC
T_comms = 2 * 2 * 2 DF / W_collective

Here we assume that our batch is sharded across X. Remember, the cost of an allreduce is the number of bytes of the array being allreduced and the bandwidth, specifically 2 * bytes / bandwidth. In the backward pass we have to perform 2 of these all reduces. This is in BF16 so we have 2 bytes per param moved hence 2DF.

For math > comms time we need B/X > C/W_collective where W_collective is either the GPU or node egress bandwidth depending on whether we are sharding within a node or across nodes. That is, we need the per-GPU token batch size to be larger than the intensity of our GPU.

- Within a node, we just need the **per GPU token bath size** > 9.9e14/450e9 = 2200
- Within a SU, or at the spine level, BS > 990e12/400e9 = 2475
- 

This is quite a bit higher than on a TPU where the number is 850 with all three axes. For instance, LLaMaA-3 which trained on 16000 H100s would need a batch size of at least 40M tokens (for reference they used 16M). DeepSeek v3 trained on 2048 H800 GPUs with a lower 300GB/s bandwidth which would need 3300 tokens per GPU, or about 6.7M batch size (they used 4M).

In theory, because these are AllReduces we are taking abuot, enabling SHARP would 2x the AllReduce bandwidth which would half all of these numbers, but in practice this benefit is closer to 30%. 

**MoE models**: 
For Mixture of Expert models where we have E experts and k experts per token, our costs change to

T_math = 4 * 2kBDF / XC
T_comms = 2 * 2 * 2 EDF / W_collective

because we have k experts performing compute and E tensors to move. This inflates the pre-GPU token batch size by a factor E/k:

B/X > E/k * C/W_collective

For example, with stats from the new OAI OSS model we get BS > 79200 which is a kind of a ridiculously high number. 

	takeaway: DP and ZeRO sharding require a per GPU batch size of about 2500 tokens to be compute bound on a h100 or b200, assuming perfect overlap and FLOPs utilization. For MoE models this increases by a factor of E/K, the ratio of total activated parameters, this is because we are only doing FLOPs on a small ratio of the total parameters. When doing a small amount of DP, such as 2-way DP, the critical batch size decreases.

#### Tensor Parallelism

**Syntax** In[B, Dy] * W_in[D, Fy] * W_out[Fy, D] -> Out [B, Dy]

One way of implementing TP is by performing an AllReduce after each matmul:

Tmp[B,  Fy] = In[B, Dy] * W_in[D, Fy] (we calculate a partial sum of the final desired product)

but we can be smarter about this, because remember we are performing two matmuls. So, instead, we can do an AllGather in the start which allows us to perform the 
matmul

In[B, D] = **AllGather**(In[B, DY]) _(on critical path)_
Tmp[B, FY] = In[B, D] *  W_in[D, FY] _(not sharded along contracting, so no comms)_

then we can perform the next matmul without and collectives as well, ending up with a partial result which we then reducescatter

Out[B, D] {Uy} = Tmp[B, FY] * W_out[FY, D]
Out[B, Dy] = **ReduceScatter**(Out[B, D] {UY}) _(on critical path)_

This saves us a decent amount on comms costs. The forward pass costs are

T_math = (2BDF/Y + 2BDF/Y) / C = 4BDF/YC
T_comms = (2BD + 2BD) / W_collective = 4BD/YW_collective

which is compute bound when 

Y < FW/C.

Within a node this gives about F/2200 or F/2475 beyond a node, this is very close to TPUs. For F=28000 like LLaMA 3 this is at about 11-way TP (or rounding down, about 8 way whichis how large a node is). That means we can
shard across up to 11 GPUs and remain compute bound, above that we are communication bound. 

	Takeaway: parallelism over an axis of size Y with feed-forward dimension F becomes communication bound when Y > F/2475, which generally constrains us to only intranode TP or at mode 2-node TP

#### Expert Parallelism

Mixture of Experts models introduce problems because the model comes with E times more model weights with only k times more FLOPs (k << E), making DP significantly harder. This can be somewhat mitigated by sharding our weights along the expert dimension i.e W_in[Ez, D, F]. To perform the MLP blocks this requires introducing 2x AllToAll collectives to dispurse our activations to the corresponding experts.

As noted above the cost of AllToAll_z->k([B, D, k]) if it spans multiple nodes is T_alltoall = 2BD (Z-8) / Z * min(8k/z, 1). This means the overall costs are

T_math = 4BkDF/ZC
T_comms = 4BD(Z-8)/WZ * min( 8k/Z, 1)

To be compute bound we need either

1) k > Z/8 with F > a (Z-8)/k 
2) Z >> k and F > 8a

where a = C/W. This gives two domains in which EP is possible, one with a small amount of expert parallelism (roughly 2 nodes) and a small F, or one with a large F and Z arbitrarily large. You'll see both cases in practice, ether small amount of EP (like DS v3 witch has a very small F and relatively small, restricted cross node EP) or models with large F, in which case we can do significant EP alongside TP.

	Takeaway if F < 8C/W, EP can span 1-2 nodes with similar cost to TP. If F > 8C/W we can do significant amount of EP, up to E nodes with relatively low cost.

#### Pipeline Parallelism

PP splits layers across nodes with an extremely low communication cost, since we are just sending the small microbatches of activations (between layers) every couple layers. Historically PP has suffered from pipeline bubbles, but with new zero-bubble pipelining approaches it is typically possible to do without. 

The overall communication cost of pipelining is tiny. With N_mb microbatches and N_stages we have

T_pp = 2BD/W_Nmb * (Nmb + N_stages - 2)
T_per_layer_comms  = 1.5 * 2BD / W N_layers

Since we divide by N_layers the comms cost are a lot smaller than other collectives. So from a communication standpoint pipelineing is basically free. But why don't we just do it then

1. Code compelxity. Pipelining does not fit nicely with automatic parallelism frameworks as other approaches. Microbatching changes the structure of the program
2. Pipelining makes DP and FSDP hard: This is probably the biggest reason. Zero 3 sharding in particular works badly since it requires us to AllGather the weights on every microbatch which doesnt work when we only have B/N tokens to amortize the AllGather cost.

#### Quiz 5: LLM rooflines

**question 1**
This means when performing DP intra node we get a roofline of B/X > 2555 to be compute bound, and for the inter node setting we get B/X > 5750. Making it harder for us to be compute bound in in the multi node regime.  For model parallelism within the node we get Y < F/2555 which is basically the same as before. 

**Question 2 [How to shard LLaMA-3 70B]:** Consider LLaMA-3 70B, training in bfloat16 with fp32 optimizer state with Adam.

1. Per parameter we need 2 bytes for weights, 8 bytes for optimizer, totalling 700GB. H100s have 80GB of DRAM which means we need at least 9 GPUs at a minimum, which is at least 2x 8xH100 nodes. 
2. This is a simple calculation. Just the total amount of required FLOPs divided by our FLOPs / second. The number of FLOPs to train a model is 6 * num params * num tokens. The available flops are 4096 * 9.9e14 * 0.45.  This would take 959 hours, or 40 days.
3. The most amount of TP we can do is given by Y < FW/C which at the node level gives Y < 11. So we can not shard across more than 11 GPUs, which essentially means we can only do 8 way model parallelism without being comms bound. 
	1. This essentially means that we will have to do 512 way pure DP. First, let us check if this is even possible because this implies that we need to be able to fit our model on a single node. Since our model, at 700GB, is sharded across 8 GPUs, our per GPU memory is 87.5GB so it wont fit! We already established this in question 1, we need to shard across at least 2 nodes to fit.
	2. With ZeRO-3 and 8-way TP we'll be doing 512-way ZeRO-3. This won't be an issue with memory because we are sharding everything aggressively across the nodes as oppposed to DP where the model and weights need to fit into each node. Our per-GPU batch size of 4e6 / 4096 = 976. This is quite low, even below our pure DP limit, and this is twice that limit because we have to more our weights. So no this is not possible to remain compute bound
	3. with PP, each model parallel shard now spans 8 nodes. As we've seen, this reduced the cost of our leaf level allgathers by 8, so the overall AllReduce/AllGather bandwidth goes from 400GB/s to 3200GB/s. The roofline is then 990e12 / 3200e9 / 309 so we should be good!