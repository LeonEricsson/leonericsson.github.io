---
layout: post
title: The Illustrated GPT-2
categories: [NLP, Transformer]
---

Blog post from Jay Alammar. Four GPT-2 models were initially released, all with their own parameter scales: GPT-2 Small (117M), GPT-2 Medium (345M), GPT-2 Large (762M) and GPT-2 Extra Large (1,542M). As we’ve seen previously the transformer model has been used in varying ways, originally introduced as a encoder-decoder architecture; it's since then been shed to implementations using either the decoder or the encoder. GPT-2 does not introduce any novel architectures but instead seeks to explore the scaling model parameters and its effects on performance. BERT, which at the time of its publication (late 2018) was the largest known language model with 300M parameters, is dwarfed by the size of the larger GPT-2 models. GPT-2 Small has the same amount of parameters (117M) as GPT-1 for ease of comparison.

Remember, GPT (Transformer-Decoder) is inherently auto-regressive and produces output tokens one at a time similar to traditional language models. BERT (Transformer-Encoder) on the other hand, is not auto-regressive and as a consequence of this it gains the capability of incorporating context on both sides of a token. GPT-2, using the decoder structure incorporates masked self-attention while BERT, using the encoder structure, uses normal self-attention. 

To get GPT-2 *talking* we can either feed it a prompt (*interactive generative sampling*) or let it ramble (*unconditional sampling*) by simply feeding it a start token \<s>. In accordance with the traditional decoder, words are embedded with a token embedder (in the case of GPT-2 it's Byte Pair Encoding) and a positional encoding. The encoded token  feeds through a classic decoder stack (stripped of the second self-attention layer commonly known as the encoder-decoder attention layer) outputting a probability distribution across the entire vocabulary. Every decoder block has its own weight matrices, four per block, which serve to: 

- Generate Query, Key and Value vectors
- Project attention heads back to model dimension
- Feed forward NN Layer #1
- Feed forward NN Layer #2

Just as with BERT, GPT-2 is trained on a large set of web pages, 8 million to be exact, with a simple casual language modelling (CLM) objective. 



Original [paper](http://jalammar.github.io/illustrated-gpt2/) (2019)
