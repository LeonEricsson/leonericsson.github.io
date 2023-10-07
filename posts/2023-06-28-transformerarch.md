---
layout: post
title: Speeding up Transformers
categories: [Transformer]
year: 2019
type: paper
---
I recently read two shorter publications both addressing speed and memory usage of vanilla transformers. I glanced through them fairly quickly and therefore I have bundled them into a single blog entry. [Generating Long Sequences with Sparse Transformers](https://arxiv.org/pdf/1904.10509.pdf) from OpenAI introduces Sparse Transformers and [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/pdf/1911.02150.pdf) is a solo paper proposing a Multi-Query Attention variant to Multi-Head attention.

## Generating Long Sequences with Sparse Transformers
Sparse Transformers are a proposed architecture to handle long sequences more efficiently. By employing a sparse attention mechanism, attention is restricted to a subset of tokens instead of computing attention for all tokens. The authors presents fixed and learned attention patterns as two strategies to achieve sparse attention. Fixed attention patterns leverage positional relationships to determine the subset of tokens attended to, while learned attention patterns allow the model to learn the relevant attention connections based on the input data. The result of this is a architecture where time and memory no longer grow quadratically with the sequence length, but rather \(O(n*\sqrt(n))\). 

## Fast Transformer Decoding: One Write-Head is All You Need
Noam, the solo author of this incredibly concise and clear paper (well done!), focuses on accelerating the decoding process in transformer models by minimizing the number of write-heads in the attention mechanism. Transformers are ubiquitous partly due to their capability of modelling inter and intra-sentence dependencies in a very efficient way. During training a lot of speed up can be achieved by parallelization across the length of the sequence. Despite this, incremental inference is often slow, due to repeated loading of the keys and value tensors. The authors tackle this problem by introducing a variant to Multi-head attention, which if you recall is multiple attention layers in parallel with different linear projections of the keys, value and queries, called Multi-query attention. Multi-query attention is identical except that the different heads share a single set of keys and values. This greatly speeds up inference and my understanding is that this method is employed in several modern LLM's.