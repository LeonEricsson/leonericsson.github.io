---
layout: post
title: Discovering faster matrix multiplication algorithms with reinforcement learning
categories: [Reinforcement Learning]
year: 2022
---

Matrix multiplication is one of computers most primitive tasks, they are everywhere and improving the efficiency of algorithms for these fundamental computations can have widespread effect. Matrix multiplications can be formulated as binary 3D tensors which define an algorithm for how to carry out such a multiplication, see figure below.

![](/images/tensormul.png)

The key here is that any decomposition of this tensor will contain another algorithm for said multiplication. The action space is enormous (more than 10^12 actions for most interesting cases) with the goal to find a low rank decomposition which will yield a more efficient algorithm. Here efficiency is defined by the number of multiplications that need to be carried out, the goal is to trade multiplications for summations as they are much cheaper on modern hardware. This paper proposes a deep reinforcement learning agent, AlphaTensor, which is trained to play a single-player game where the objective is to find tensor decompositions within a finite factor space. 


Original [paper](https://www.nature.com/articles/s41586-022-05172-4) (2022)
