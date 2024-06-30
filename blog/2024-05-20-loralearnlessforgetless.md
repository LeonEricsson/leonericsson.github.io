---
layout: post
title: "LoRA Learns Less and Forgets Less"
categories: []
year: 2024
type: paper
author: Biderman
exturl: https://arxiv.org/pdf/2405.09673
---
lora learns less of a target domain than full fine-tuning (fft) (evaluated on code and math; humaneval and gsm8k) but this difference is not consistent across domain or training budget.

naturally, lora forgets less of the source domain. lora seemingly represents a smaller perturbance of the original weights.

given the above; does LoRA and fft represent different tradeoffs between learning and forgetting? no. they occupy the same pareto curve. i would have liked to see a compute comparison at this stage, to complement the trade-off analysis. depending on your desired target domain performance it seems lora is much more compute efficient.

fft on code and math does not learn low-rank perturbations. the implied rank (to explain 90% of the variance of a $4096 \times 4096$ matrix) for attention modules is in the 1000-2000 range, and 2000-2800 for mlp. this is *a lot* more than normal LoRA rank values.

practical considerations: lora should be used for instruction fine-tuning, not continued pre-training. one should identify the highest lr that enables stable training, and target modules in order: all > mlp > attention.