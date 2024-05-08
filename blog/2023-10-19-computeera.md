---
layout: post
title: "distributed training for the gpu-poor"
categories: []
year: 2023
type: blog
---

In today's computational arms race, the distinction between the "haves" and "have-nots" couldn't be starker. On one end of the spectrum, we have organizations that are endowed with colossal computing resources. Research centers at companies like 🏢 OpenAI, 🌐 Google, and 📱 Meta that command arsenals of 20,000+ H100 GPUs. These behemoths are not just awash in silicon; they're using their GPU prowess as an active recruiting strategy, luring the best and brightest minds in machine learning.

On the opposite end, there exists a vast sea of startups and individual researchers termed as "GPU-poor." These entities are continuously engrossed in cobbling together models using a limited set of resources, often overlooking the far-reaching implications of computational inefficiency. A lack of understanding of effective model trade-offs has led to fruitless endeavors—focusing on leaderboard-style benchmarks and broken evaluation methods. They're more concerned about fine-tuning models with insufficient VRAM than about maximizing efficiency at scale.

It's a concerning trend among GPU-poor that has led to a misguided allocation of resources. The benchmarks have little bearing in the real-world and there is no room for these models in a world inundated by millions of H100. In such a landscape, obsessing over benchmarks is akin to rearranging deck chairs on the Titanic. Instead, understanding how to navigate the trade-offs in model performance, token-to-token latency, and compute requirements can make a world of difference, particularly for the underdogs. If you're in this for the long haul, it's time to shift gears and focus on what truly matters—efficiency, scalability, and the effective use of whatever compute resources you can muster.

## Transformer math during training

#### Scaling Laws
The basic equation that governs the cost of training a Transformer model is given by

$$ C \approx \tau T = 6PD$$

in total floating point operations. Naturally, the total compute is a product of the system throughput $\tau$ and the training time $T$ but thanks to valuable work from [OpenAI](/posts/2023-05-25-scalinglaws.md) and [DeepMind](/posts/2023-08-01-chinchilla.md) we now know it scales proportionally with the number of parameters $P$ and the number of tokens $D$. Since Chinchilla we've also become accustomed to the _compute optimal_ trade-off between $D$ and $P$ which should satisfy

$$ D = 20P.$$

However, although optimal, it is often unwise to train models on "too" few tokens. ElethurAI state that: "_We do not recommend training a LLM for less than 200B tokens. Although this is “chinchilla optimal” for many models, the resulting models are typically quite poor. For almost all applications, we recommend determining what inference cost is acceptable for your usecase and training the largest model you can to stay under that inference cost for as many tokens as you can._"


#### Memory costs during training
Unfortunately, training is always going to cost more than inference. During inference you only have to store model weights ($\times 1.2$ in overhead) but for training you also need the optimizer state and the gradients on device memory. For vanilla AdamW that's 2 bytes for a copy, 4 bytes for momentum and 4 bytes for variance - per parameter for a total of 10 bytes/parameter. Gradients are commonly stored bf16 meaning 2 bytes/parameter.

Lastly, we need to store the activations of our model. Now, storing all the activations simply takes up too much memory so activation recomputation/checkpointing has become the norm when training large models. Activation recomputation/checkpointing works by recomputing activations of certain layers instead of storing them in GPU memory. The compute / memory trade-off achieved by this method is determined by which recomputation scheme you use but to make it easy for us lets assume we recompute all activations meaning the memory requirements are

$$ memory_{activations} = 2 * sbhL$$

where $s$ is the sequence length, $b$ the batch size, $h$ the hidden dimension and $L$ the number of layers. The total memory required during training is the sum of model memory (2 bytes per parameter for bf16), optimizer memory, gradient memory and activation memory. Let's illustrate this with an example; Llama 2-13B has the following specs:

- Model parameters $P$: 13B
- Sequence length ($s$): 4k tokens
- Batch size ($b$): 4M
- Hidden dimension ($h$): 5120
- Number of layers ($L$): 40

1. **Model Memory**: $13\text{e}9 \times 2 = 26\text{GB}$
2. **Optimizer Memory**: $13\text{e}3 \times 10 = 130\text{GB}$
3. **Gradient Memory**: $13\text{e}9 \times 2 = 26 \text{GB}$

Unfortunately the activation memory is a bit tricky to calculate because nobody does full recomputation, it's way to memory expensive. Glossing over that part we note that for a model of "only" 13B parameters we end up needing **at least** 182GB of memory! 

## You can't espace parallelism

Even if you had access to a H100 with 80GB of vRAM you still have to resort to distributed training and thats where strategies such as Data, Model, Pipeline and Tensor Parallelism come into play. Naturally these can be combined for better throughput gains, but let's take a closer look at some improvements made to data parallelism: Zero Redundancy Optimizer and the closely related Fully Sharded Data-Parallel strategies.

### ZeRO
In Data parallelism, a traditional batch is split into mini-batches and distributed amongst a number of workers. Each worker calculates the gradients with respect to their mini-batch and the gradients are then averaged across all workers before the model is updated. In its most naive implementation (PyTorch DPP), each worker holds a copy of the model weights, optimizer state and gradients. Apart from memory costs, this approach incurs a communication cost of $2\Phi$, where $\Phi$ is the total number of computed gradients.

ZeRO Stage 1 introduces optimizer state partitioning across the GPU workers with gradients and model weights still being replicated. In the same way as before, gradients are communicated to each worker which all update their optimizer states. Thankfully, the optimizer state (at least for Adam) has no dependency across different slices of weights so each worker still only needs the gradients to update their partition of the optimizer state. This means we can reduce memory consumption without effecting communication volume. 

If you are very observant, you may have asked yourself why we are communicating the average gradients across all workers if each worker only updates its own partition of the optimizer state. Well, we shouldn't. ZeRO Stage 2 introduces gradient partitioning to avoid this exact thing. Gradients are sharded across workers and we don't need to communicate these gradients because the optimizer state is updated from our own gradient partition. This means we can have even more memory savings with the same communication volume! Note however that both Stage 1 and Stage 2 assume that the entire model fits on 1 GPU.

Finally, we reach the holy trinity of optimizer, gradient **and** parameter partitioning in ZeRO Stage 3. This is where things, in my opinion, start to get a bit more difficult to grasp. Recall that up until this point: each worker forwards a mini-batch through the entire model(which it has stored on device), calculates the gradients of this mini-batch and updates its partition of the optimizer state. The optimizer state is then communicated across all workers such that each worker can update its copy of the model weights w.r.t the entire batch. Now, in ZeRO Stage 3 the model layers are sliced horizontally with each worker storing part of each layer weight tensor. During the forward and backward pass, the activations are communicated inter-GPU leading to a total communication cost of $3\Phi$. While the process may seem excessive, we've managed to cut down memory consumption by the number of GPU workers $N$ for a communication cost only 1.5x larger than naive DP. That's pretty wild! ZeRO Stage 3 also means that we can now fit models of arbitrary size as long as their are sufficient GPU workers. 

ZeRO has continued to see optimizations since the release of Stage 3. ZeRO-Offload enabled offloading of optimizer and gradient states to CPU memory, ZeRO-Infinity improved these offloading techniques by offloading to disk (NVMe memory) and ZeRO++ introduced model weight quantization, hierarchical partitioning and quantized gradients. 

### How can I use it?
The neat thing about ZeRO is that its all done in the data parallelism regime, imposing very little work on the person who want to use it. ZeRO is available, in it's entirety, through the Microsoft DeepSpeed library and FSDP is a part of PyTorch itself.

But what about Tensor Parallelism and Pipeline Parallelism you might be asking? Well, because both of these require architecture changes and/or changes in the forward pass of the model there simply isn't a one-fits all solution out there yet. If you really do want PP and TP, the best option for now seems to be to use Megatron-LM and stick to the models they support (BERT, GPT-2, T5, Llama). You can also make use of ZeRO-powered DP + DeepSpeed PP + Megatron TP in Megatron-DeepSpeed, but only for training models based on BERT, GPT-2 and T5. From Stas Bekman: *TP+PP+DP is more efficient than ZeRO but is extremely difficult to develop with and troubleshoot as it requires massive changes to the modelling code.* 

