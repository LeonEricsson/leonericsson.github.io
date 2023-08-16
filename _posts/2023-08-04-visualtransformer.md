---
layout: post
title: "An image is worth 16x16 words: Transformers for image recognition at scale"
categories: [Computer Vision, Transformers]
year: 2020
type: paper
author: Dosovitskiy
exturl: https://arxiv.org/pdf/2010.11929.pdf
---
Given the widespread attention (no pun intended) and success of the transformer architecture across the NLP field, multiple works have tried combining convolutional architectures with attention to push the computer vision field to new heights. The problem often faced with however is that a naive application of self-attention would require a global receptive field for every pixel and with a quadratic cost in the number of pixels this doesn't scale well on modern hardware. So, people have made sacrifices and modifications to adjust for this but as of this papers publication no convincing results had been achieved and classic ResNet-like architectures were still state of the art. The authors of this paper however try to take as simple approach as possible using the standard Transformer with as little as modifications as possible. Given sufficient pre-training the authors find that this scales well and it opens up a lot of intriguing paths for the future. 

## Vision Transformer
As we know by now, a standard Transformer receives as input a 1D sequence of token embeddings. But an image is 2D, even 3D given the three color channels so how do we transform our image into a proper sequence? This is where the authors main contributions lie. First an image is split into fixed-size patches and then each patch is flattened to be projected by a trainable linear projection into a sequence N x D where N is the number of *patch embeddings* (equivalent to token embeddings) and D is the model size used throughout the transformer layers. Position embeddings are added to the patch embeddings to retain positional information. This sequence is used as input to a number of Transformer encoders which consists of alternating layers of multiheaded self-attention and MLP blocks. LayerNorm is applied before every block and residual connections after every block. MLP blocks use GELU non-linearity. 

## Findings
Given sufficient pre-training the vision transformer outperforms state of the art ResNet-based architectures across almost all evaluated tasks. A detailed study of performance versus total pre-training compute clearly shows how Vision Transformers dominate ResNets on the performance/compute trade-off. ViT uses approximately 2-4x less compute to attain the same performance and while Hybrid models outperform ViT on small compute budgets this discrepancy reduces for larger computes. Furthermore, results seem to indicate that the inductive bias inherent to ResNet's convolutional architecture is highly beneficial when working with smaller datasets but, given enough data, learning these patterns directly from the data is sufficient, perhaps even beneficial. Note that ViTs have much less image-specific inductive bias as compared to CNNs. In CNNs, locality, two-dimensional neighborhood structure, and translation equivariance are
baked into each layer throughout the whole model. In ViT, only MLP layers are local and translationally equivariant, while the self-attention layers are global.

## Thoughts
I feel like this makes a case for further research into this architecture for CV tasks and I would be interested to see how this translates to things such as detection and segmentation which is not experimented with here. Importantly ViTs, similar to LLMs require a large amount of pre-training as compared to alternative architectures, but given this they seem to be extremely efficient in their performance. Interested to see how this work translates into multi-modal work in the future. 