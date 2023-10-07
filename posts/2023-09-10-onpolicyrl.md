---
layout: post
title: "What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study"
categories: [Reinforcement Learning]
year: 2020
type: paper
author: Andrychowicz
exturl: https://arxiv.org/pdf/2006.05990.pdf
---
Reinforcement Learning has seen an upswing over the past few years especially when talking about the subfaction of on-policy RL algorithms. Historically, RL has seen successful application in discretized, simple state space environments but there has always been struggle when moving into a continous domain with an increasing state space complexity. The convergence is too slow, learning is un-parallelizable and compute is insufficient. On-policy reinforcement learning has emerged as a viable option in light of these complications. However, RL still remains a fragile practice, while algorithms are conceptually simple their state of the art implementations take numerous low- and high-level design choices that strongly affect the performance of the resulting agents. Design choices such as initialization, hyperparameter tuning, network design etc remain crucial and are often glossed over in published descriptions. This paper presents a large emperical study of such design choices, evaluating over 250,000 agents in five continous control environments. They present concrete advice for practitioners of on-policy RL and I'm going to share some of their insights here today. Besides the actual experiments and insights there isn't much to share here, if you are interested in more details about their setup please check out the original paper.

## Policy losses
The first investigation involves different policy loses such as vanilla policy gradient, PPO, AWR and V-trace. The goal here is to better understand the importance of the policy loss function in the on-policy setting. PPO proves hands down to be the best performing losses, beating out others in 4 out of 5 test environments. PPO clipping is also evaluated and seems to be dependent on environment but the recommendation is to start with a threshold of 0.25 and test lower/higher if possible.  

## Network architecture
A number of design choices related to network architecture such as value and policy networks, activation functions, network structure and size, and initialization. The concrete findings and recommendations are listed below.
- Seperate value and policy networks perform best
- Optimal policy network width depends on problem complexity so this needs to be tuned.
- Value network seems to benefit from wider networks. No observed downside.
- Initial policy has very high impact on training performance! It is recommended to initialize the last policy layer with 100x smaller weights, use softplus to transform network output into action standard devation and add a negative offset to its input to decresase the initial standard deviation of actions.
- Tanh is the best performning activation function, ReLU is the worst.

## Normalization
An investigation of different normalization techiques: observation normalization, value function normalization, per-minibatch advantage normalization, as well as gradient and observation clipping finds that input (observation) normalization is crucial for good performance.  

## Timestep handling
Traditional RL hyperparameters such as discount factor, frame skip, and how episode termination due to timestep limits are handled are investigated. Authors find that discount factor is one of the most important hyperparameters and should be tuned per environment (start with 0.99) and that frame skip should be tried if possible.

## Optimizers
Adam is compared with RMSprop (as well as their hyperparameters) and the findings are that Adam should be used with a momentum of 0.9 and a tuned learning rate with 0.0003 as a safe default. Linearly decaying the learning rate may improve performance but is of secondary importance. 
