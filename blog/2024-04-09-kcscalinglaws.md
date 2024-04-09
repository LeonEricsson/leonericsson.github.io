---
layout: post
title: "Making sense of Floating Points"
categories: []
year: 2024
type: paper
author: Allen-Zhu
exturl: https://arxiv.org/pdf/2404.05405.pdf
---

---

A model with 100M parameters storing 220M bits of knowledge has a capacity ratio of 2.2 bits per parameter. This *knowledge* does not refer to word-for-word memorization but rather knowledge that is extractable and applicable to downstream tasks.

---

**Statement 1.** GPT-2 consistently achieves a 2bit/param capacity ratio across all data settings after sufficient training. This holds across a **wide** variety of model configurations (size, depth, width, etc..)

**Statement 2.** Most LLM architectures achieve this 2bit/param ratio if *sufficiently* trained, even with MLP layers removed, this is universal. Sufficiently trained in this context refers to each knowledge piece being visited 1000 times during training, referred to as **1000-exposure**. When exposure falls to 100 times, GPT-2 is *undertrained* and achieves a capacity ratio of 1bit/param.

**Statement 3.** In the **100-exposure** setting, LLaMa and Mistral's capacity ratio is 1.3x lower than GPT2's. The authors point to GatedMLP as the culprit.

**Statement 4.** Quantizing models to int8 does not compromise knowledge capacity; however, at int4, capacity falls to 0.7 bit/param

**Statement 5.** MoE models, even at 32 experts, only reduce 1.3x in capacity compared to base scaling laws, despite only using **8%** (!!) of the total parameters during inference.

**Statement 6.** "Junk" data such as CC significantly reduces model capacity. As an example, with a 1:7 ratio of useful-to-junk training tokens, capacity for useful knowledge *loses by a factor of 20x* even when useful knowledge is exposed 100 times.

**Statement 7.** An effective mitigation to the above is to prepend a special token to all useful knowledge. This is akin to adding a domain name like wikipedia.org at the start of every Wikipedia paragraph; the model autonomously identifies high-quality data without prior knowledge of valuable domains. In the example above, the loss factor improves from 20x to 2x.