---
layout: post
title: "An observation on Generalization"
categories: [Deep Learning]
year: 2023
type: presentation
author: Sutskever
exturl: https://www.youtube.com/watch?v=AKMuA_TVz3A&ab_channel=SimonsInstitute
---
A couple of days ago OpenAI's chief scientist, Ilya Sutskever, held a talk at the Simon Institute and while I was listening I decided to jot down a few pointers that I found interesting. Hope you find something of value but I understand a post like this is more personal than educational.

- Supervised learning is theoretically sound and conceptually trivial. We have formal proofs that given more training data than degrees of freedom and IID test data -> Low training error = low test error. 

- Unsupervised learning on the other hand is confusing and very much unlike supervised learning. For instance we optimize for one objective (e.g. next word prediction) but we care about a different objective. Intuitively we can reason that learning features from the input distribution can somehow tell us useful things about the task but is there any way to reason about this mathematically?

- Compression provides a way for us to reason about unsupervised learning in such a way. Compression is well-known to be synonymous with prediction - a good predictor can be made into a equally good compressor and vice versa. An easy example of this is a character encoding scheme where more frequently appearing characters are encoded with fewer bits. Just by reasoning about the likelihood of the next character, e.g predicting it, we are able to reduce the number of bits needed. 

- Given two datasets X and Y, and a decent compression algorithm C() a join compression of X and Y C(concat(X,Y)) will intuitively use patterns that exist in X to help compress Y (and vice versa). This line of thought is the same for prediction, but it makes more intuitive sense when talking about comprediction. Along the same line of reasoning we find that no compression of the joint data should be larger than the independent compression of the two datasets - C(concat(X,Y)) < C(X) + C(Y). Any kind of additional compression gained by concatenation is a result of shared structure in the data. Now if we imagine Y to be a supervised task and X to be a unsupervised task we have some kind of mathematical reasoning behind the benefits gained through unsupervised pre-training. 

