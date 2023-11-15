---
layout: post
title: "Rethinking Benchmark and Contamination for Language Models with Rephrased Samples"
categories: [NLP]
year: 2023
type: paper
author: Yang
exturl: https://arxiv.org/pdf/2311.04850.pdf
---

Introducing Llama-rephraser: a 13B model that is competitive with GPT-4 across major benchmarks (MMLU, GSK-8K, HumanEval). The authors follow OpenAI's decontamination method of n-gram overlap, finding no evidence of test-train data contamination. What special dataset was this trained on? What is the new attention mechanism that enabled such performance? Nothing that fancy, just paraphrase or translate the test set and you'll turn out with a 13B LLM smart enough to generalize beyond such variations and achieve drastically higher benchmark performance! /s

On a more serious note, data contamination is a recognized issue of LLM benchmarking and it remains a open problem. There is a reason why the community is quick to disregard reported performance benchmarks in favor of the more subjective _feel_ of a model. This paper studies a particular issue of common decontamination methods and proposes a new method to counteract said issue.

# Contamination methods

Contamination occurs when test set information is leaked in the training set, resulting in a overly optimistic estimate of the model's performance. The commonly used contamination methods include:

**N-gram overlap**. Used extensively at OpenAI, n-gram overlap defines contamination if $n$ tokens/words/grams overlap. GPT-3 paper defines a 13-gram overlap as contamination, and the GPT-4 report uses 50-character overlap.
**Embedding similarity search**. This method transcends mere textual overlap by using transformer-generated embeddings to capture prompt semantics. While it provides a more nuanced view than n-gram overlap, setting the appropriate similarity threshold is a challenge.
**Decoding matching**. Useful when training data isn’t accessible, this method banks on the idea that a model trained on contaminated data is likelier to complete a known test prompt, which has its own set of limitations.

# Rephrasing

This paper's critical examination focuses on the impact of including rephrased test samples in training sets. The "RephraseLLM" dataset, crafted for this study, consists of such samples. These are strategically altered to maintain their original meaning while evading detection by standard contamination methods.

The rephrasing process involves synonym substitution and restructuring sentence order. An LLM aids in this rephrasing, ensuring the altered samples remain undetectable by traditional n-gram checks. Additionally, test samples are translated into different languages, adding another layer of complexity to "RephraseLLM".

## Experiments

The experiments with Llama-based models trained on the RephraseLLM dataset reveal a substantial performance increase on benchmarks, challenging the efficacy of current contamination detection methods. This result highlights the need for a more robust approach to identify and mitigate the influence of subtly altered data on model performance.

# A new decontamination method

Given the inability for todays contamination methods to find rephrased contaminations that heavily skew model benchmark performance, the authors propose a new decontaminator: **LLM Decontaminator**.

This two-step process first identifies the top-k training items with the highest similarity using embedding similarity search. Then, it evaluates each pair with an advanced LLM like GPT-4 to determine if they are essentially the same. This method proves more effective in identifying test-training overlaps, including those undetected by previous methods, demonstrating its potential as a more reliable tool for decontaminating LLM training sets.
