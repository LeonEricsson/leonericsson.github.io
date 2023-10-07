---
layout: post
title: "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
categories: [NLP, Transformer]
year: 2019
type: paper
author: Dai
exturl: https://arxiv.org/pdf/1901.02860.pdf
---

We dive into yet another transformer based architecture, this time with a paper tackling the fixed-length context of the original *vanilla* transformer in addition to proposing a simple but more effective relative positional encoding formulation. The introduction of this paper suggests that transformers, as an architecture, still hasn't completely **claimed** the field and research is still divided between it and RNN/LSTM. 

## Context Fragmentation
The authors coin context fragmentation as in inherent problem of the original vanilla transformer and a prevalent oversight in a lot of current research. The problem arises from the separated fixed-length segments that the language models are trained on where no information flows across the sequences, limiting contextual understanding above the sequence length. This makes prediction of the first tokens difficult and understanding of longer sequences impossible. This means that despite self-attention being less effected by vanishing gradient and more effective in computation it can't leverage this optimization advantage.

## Segment-level recurrence
The authors proposes a modification to the Transformer architecture by introducing a recurrence mechanism to address its limitations in modeling long-term dependencies in sequences. The hidden state sequence from the previous segment is cached and reused as an extended context for the next segment. This creates a segment-level recurrence in the hidden states and allows for modeling of longer-term dependencies, avoiding context fragmentation. Additionally, the recurrence scheme results in significantly faster evaluation and the ability to cache multiple previous segments as memory to be used during evaluation.

## Relative Positional Encoding
Due to hidden representations being cached and reused for subsequent segments calculations, the absolute positional encoding in the vanilla transformer is no longer valid. Previously the position was only calculated with respect to the segment but instead the authors now propose a relative positional encoding scheme that enables the model to learn from previous tokens outside of the current segment. 

## Final thoughts
It's interesting to see how the concepts of RNNs find themselves useful in the transformer architecture and even more so if this continues to be a trend moving forward. The fact that this recurrence change improved evaluation speed to such a degree was also cool to see. 






