---
layout: post
title: "Efficient Training for the GPU-poor"
categories: []
year: 2023
type: blog
---
The last post was running quite long so I decided to move the second part into this post where I want to discuss some of the [efficient training and fine-tuning](https://sumanthrh.com/post/distributed-and-efficient-finetuning/) techniques that are available for researchers today. These techniques focus on what we can do to improve throughput, efficacy and memory consumption locally inside one GPU. 

## Mixed Precision
Now a days, this is a given for large model training. Weights, activations and gradients are all stored in half-precision format while you have a "master copy" of the weights in FP32 precision. The most prominent half-precision format is Brain Float 16 developed by Google Brain

## Flash Attention 
is a IO-aware, memory-efficient and exact attention algorithm enabling immense speedups. It leverages kernelized attention and a hierarchical decomposition of the attention matrix to achieve linear time complexity for long sequences. This is a departure from the traditional attention mechanism which computes pairwise interactions in quadratic time. FlashAttention avoids full matrix multiplication and instead uses kernel-based approximations to speed up the computations, preserving the model's expressiveness and accuracy while being computationally efficient. It's ridiculously smart given how straight forward the actual implementation is. The only drawback is that FlashAttention is hardware dependent and is currently "only" supported on Ampere, Ada and Hopper NVIDIA GPUs with bf16 precision. Very soon FlashAttention will become a part of PyTorch's internals.

## Gradient Checkpointing
I talked about this in the previous post, instead of retaining all of the activations in memory we can recalculate them when necessary during the backward pass. This is a compute/memory trade-off, as always. A good rule of thumb from HuggingFace is that gradient checkpointing slows down training by 20%.

## Gradient Accumulation
If you are memory bound but want to reduce your effective batch size at the drop of some throughput, you can skip your optimizer step for X amount of training steps. This means you can double the batch size without any increased memory usage and by just accumulating the resulting gradients in between optimizer updates. 

## Parameter-Efficient Fine-Tuning (PEFT)
PEFT methods aim to reduce the memory requirements during finetuning, by freezing most of the model weights and having a subset/ a small number of additional parameters as trainable. The most popular PEFT method is LoRA, where you finetune a low-rank version of weight updates to your model parameters. Another effective PEFT method is IA, which injects trainable vectors into key, value and feedfoward layers in a transformer-based architecture. With both LoRA and IA, the added weights/vectors can be merged with the base weights, meaning that, at inference time, there are no additional computations (addition/multiplication). The downside is that performance can be lesser than when you perform full finetuning. This however, has rapidly changed, and LoRA-based approaches can infact match full fine-tuning performance, if you add trainable weights to ALL linear layers (See QLoRA).

## Quantization
I've talked about post-training quantization in a [previous](/posts/2023-09-17-llminference.md) post but there is also **quantization-aware training*. This is where quantized weights and weights are apart of the trained model. QLoRA is an example of a quantization-aware training technique. The main idea with QLoRA is that it quantizes the base, pretrained model weights to 8/4 bits and then trains additional LoRA parameters in floating-point half/full precision. This is a very powerful strategy, enabling finetuning of 60B+ parameter models on a single GPU with 48GB vRAM. The full QLoRA paper is worth reading. Beyond the fact that their approach enabled training a 65B model on 1 consumer GPU (this was the largest open source language model at the time), the paper also showed that LoRA-based training can match full fine-tuning

## Practical Guidelines
To end off, let's list of a couple of practical guidelines for people interested in training large transformer based models.

- BF16/ FP16 by default.
- Use Flash Attention when supported
- LoRA for fine-tuning
- Use gradient checkpointing when you can't use Flash Attention
- Use an efficient sampler in your dataloader, like the multi-pack sampler.
- **If you have multiple GPUs, always try BF16 + LoRA + Gradient Checkpointing + DeepSpeed ZeRO 3 first.**
- DS Zero 3 is dependent on sufficient inter-GPU communication
- When you’ve got a new infra setup and wish to try out DeepSpeed, you should definitely use [DeepSpeed’s memory estimators](https://deepspeed.readthedocs.io/en/latest/memory.html)
- Use quantization when you have very limited GPU memory.
- In a small-scale multi-node setup, with a few nodes, the best option seems to be DeepSpeed ZeRO-3 with hierarching partitioning enabled (or FSDP with hybrid sharding). If you’ve got Infiniband interconnect, you can mostly use plain DeepSpeed ZeRO-3 and push for larger model sizes as well.
- Gradient accumulation should be used if you’re still short on batch size after all the above optimizations. Training times with gradient accumulation can be faster with large models and multi-GPU/ multi-node settings.
- Finally, when you do start training, monitor htop to check on RAM usage (sometimes RAM OOM can be an issue), along with nvidia-smi to make sure GPUs aren’t bottlenecked by data preprocessing (you should aim for close to 100% volatile GPU utilization, even if GPU memory usage is lesser).
