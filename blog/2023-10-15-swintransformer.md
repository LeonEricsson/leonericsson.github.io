---
layout: post
title: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
categories: [Computer Vision, Transformer]
year: 2021
type: paper
author: Liu
exturl: https://arxiv.org/pdf/2103.14030.pdf
---

Ever since the Transformer became the defacto architecture for natural language processing, there's been a growing interest in its application for other modalities. Fundamentally, vision isn't too different from text. Yes, we're working in 2-dimensions and with pixels as opposed to sequences of words, but we can represent an image linearly through projection and position can be encoded similarly to how its done already in NLP Transformer. It turns out the problem isn't the image itself but rather the scale of images and the inefficiency of global attention.

## Addressing the Limitations of the Vision Transformer

I delved into these challenges in my earlier piece on the [Vision Transformer (ViT)](/posts/2023-08-04-visualtransformer.md). The beauty of ViT lies in its ability to leverage the quintessential transformer architecture. By segmenting an image into a set number of windows, ViT manages to side-step the quadratic complexity tied to global attention. Yet, it demands extensive pre-training and its complexity remains tied to the quadratic rise with image size. Designed primarily for image classification, a gap remains for a general-purpose backbone that could potentially replace stalwarts like ResNet—a mainstay in computer vision for years. Enter the Swin Transformer, which seeks to redefine this landscape, integrating the best from classic architectures like ResNet and VGG.

## Swin Transformer

The Swin Transformer employs a dual-method approach to dissect an image. Echoing ViT, it first divides an image into distinct non-overlapping patches. Each patch, a combined set of RGB pixel values, is then linearly embedded to produce a token, much like tokens in traditional NLP.

As these patches journey through the network, they alternate between Swin Transformer Blocks (modified self-attention modules) and patch merging phases. During the merging stages, adjacent patches are fused and then projected into a feature space with doubled dimensionality. For instance, starting from a set of patches represented in dimensions of \( \frac{H}{4} \times \frac{W}{4} \), after merging, the dimensionality morphs to \( \frac{H}{8} \times \frac{W}{8} \), but with channel depth doubled. For aficionados of classic convolutional networks like ResNet, this hierarchical paradigm will ring a bell. Indeed, this structure stands as one of the Swin Transformer's cornerstones. And as we progress through the layers, the final output adopts a dimension of \( \frac{H}{32} \times \frac{W}{32} \times 8C \). Crucially, the Swin Transformer doesn't impose a preset division for the patches, allowing adaptive scaling of input images. With this groundwork laid, it's time to delve deeper into the nuances of the Swin Transformer Block.

## Swin Transformer Block

Traditional global attention results in a formidable computational complexity—quadratic in relation to token count—making it a poor fit for many vision tasks. In the Swin Transformer, this challenge is met head-on by evenly breaking an image into non-overlapping windows, each encapsulating a fixed number of \( M \times M \) patches (with \( M = 4 \) as per the original paper). Within each window, tokens are restricted to attending to their immediate neighbors. With this fixed window size, computational demands are capped, even as we cater to larger images. Consequently, computational requirements scale linearly with image size, as opposed to the burdensome quadratic scaling.

A notable limitation of this approach, however, is the absence of inter-window connections. To remedy this, the Swin Transformer introduces a shifted window partitioning strategy. By alternating partitioning configurations across consecutive blocks, cross-window connections are established without sacrificing the computational benefits of distinct windows.

## Final Thoughts, Impact, and Future

The Swin Transformer is genuinely exciting. Marrying the best of traditional architectures with the adaptability of transformers, it's like getting a fresh take on an old favorite. Could it replace veterans like ResNet? Maybe. But what's clear is its design prioritizes scalability and efficiency, which are always a big win in the deep learning world. Looking ahead, I think we're going to see a lot more from Swin Transformer and similar models. They've definitely stirred the pot, and I'm eager to see where this leads in the landscape of computer vision.
