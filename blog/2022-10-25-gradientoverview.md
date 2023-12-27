---
layout: post
title: "An overview of gradient descent optimization algorithms"
categories: [Deep Learning]
year: 2017
type: paper
author: Ruder
exturl: https://arxiv.org/pdf/1609.04747.pdf
---

Vanilla batch gradient descent computes the gradient of the cost function w.r.t the parameters for the entire training set. Stochastic gradient descent performs parameter updates for each training example which does away with some redundancy from vanilla batch, it also introduces some fluctuations as SGD performs frequent updates with high variance which enables it to jump to new potentially better local minima. SGD requires a slowly decreasing learning rate to converge otherwise it will keep skipping around. Mini-batch gradient descent is the best of both worlds, it reduces the variance of the parameter updates leading to a more stable convergence (more accurate representation of the entire data set) c) highly effective compared to SGD. Modern optimization methods use the following to speed up convergence

- Momentum term which increases for dimensions whose gradient points in the same direction.
- Adaptive learning rate, performs larger updates for infrequent and smaller updates for frequent parameters. No need for manual tuning of learning rate

