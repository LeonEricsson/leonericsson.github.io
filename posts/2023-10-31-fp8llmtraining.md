---
layout: post
title: "FP8-LM: Training FP8 Large Language Models"
categories: [NLP]
year: 2023
type: paper
author: Peng
exturl: https://arxiv.org/pdf/2310.18313.pdf
---
Quantization has become a hacker's favorite, acting as a democratizing force in the ever **scaling** landscape that is modern deep learning. HuggingFace is blowing up with quantized versions of models, readily available for people to try out all thanks to enthusiasts such as [TheBloke](https://huggingface.co/TheBloke). This kind of work, known as *Post-Training Quantization* has proved very effective. There is also *Quantization-aware training* which integrates quantization directly into the model training recipe. In the original sense of the term, this meant to train a model from the start using quantized weights and activations, for use later during inference. QLoRA falls into this category, by quantizing the base weights of a pretrained model to 8/4 bits then train additional LoRA parameters in half-precision on-top of the base. Anyway, I'm getting off topic; the point is that quantization works but we've always had to accept the performance / memory trade-off. Well, maybe not for much longer? This fresh paper from Microsoft Research proposes a brand new FP8 mixed precision training framework, unlocking 8-bit weights, gradients, optimizer and distributed training. 

## Floating Point 8-bit 

What is the floating point 8-bit format?

## Okay, so just cast everything to FP8?
What are the difficulties with FP8?

data underflow overflow, quantization errors arrising from the narrow dynamic range. leading to loss spikes and NaN.

FP8 optimization strategy: FP8 communication, FP8 optimizer, FP8 distributed training

## Why does even work in the first place?

- **Quantization:** The values in neural networks, especially weights, don't cover a broad range uniformly. Many values might be close to zero. Quantization techniques map this non-uniform distribution to a lower precision representation effectively.
- **Noise Tolerance:** Neural networks, especially deep ones, have shown resilience to noise. In fact, adding noise to activations, weights, or gradients is a common regularization technique (like dropout). Reduced precision arithmetic can introduce a form of noise, and if the network is large enough, it might be able to tolerate or even benefit from it.
- **Mixed Precision Training:** It's possible to use FP8 for certain parts of the training process and higher precision for others. For instance, activations might be in FP8, but certain gradient calculations might use FP16 or FP32.


