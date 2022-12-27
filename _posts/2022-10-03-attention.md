---
layout: post
title: Attention is all you need
categories: [NLP]
---

Authors present a novel architecture called the Transformer using only self-attention mechanisms to compute representations of the input and output. Previous work has used the attention mechanism but in conjunction with recurrent neural networks. Recurrent models have a few inherent problems that forced the revolution in this paper. They can be difficult to train due to their long dependency structures often resulting in gradient issues, they have computational issues on long sequences as their dependency graph is narrow and deep thus preventing parallelization and traditionally RNNs based on hidden state vectors had trouble modeling long-term dependencies. In transformers the number of operations required to relate signals from two arbitrary input or output positions is constant. A huge improvement to previous work where it’s been logarithmic at best.

The attention function maps queries + key-value pairs to an output where all elements are vectors. A compatibility function measures how well the query matches each key from the input sequence. In this case the compatibility function is implemented as a dot product between the query and the keys, scaled by 1/d (d is dimension of query and key vector) and a softmax applied to obtain the weights on the specific values. An alternative to dot product attention is additive attention which uses a feed forward network to compute the compatibility function. For large dimension d, the dot product grows large pushing the softmax into a saturated state, this is the reason for the 1/d scaling factor. 

Input tokens are embedded ahead of the encoder/decoder stacks using layers sharing weights converting them to vectors of dimensions d_model. Since the architecture doesn’t have recurrence or convolution there needs to be an injection of positional information (relative or absolute) of the tokens in the sequence. 

Original [paper](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) (2017)
