---
layout: post
title: Human-level control through deep reinforcement learning 
categories: [Reinforcement Learning]
---

Breakthrough paper for Deep RL where Mnih et. al managed to surpass performance of all previous algorithms and achieve a level comparable to professional humans across a set of 49 Atari 2600 games using the same algorithm, network and hyperparameters. Prior to this paper, reinforcement learning was limited to domains where useful features could be handcrafted or domains with fully observable, low dimensional state spaces. But, thanks to a clever invention called the deep Q-network (DQN), reinforcement learning has modernized and is now present in many multimodal systems. There are a couple of fascinating takeaways I want to touch on from this paper.

Firstly, the DQN algorithm uses a neural network to approximate the Q-function, which gives the expected future reward for each action in a given state. The Q-function is updated using the Bellman equation, which defines the relationship between the expected reward for a given action and the expected rewards of the subsequent actions that the agent will take.

Secondly, the DQN algorithm uses a target network, which is a copy of the main Q-function network that is used to stabilize the learning process. The target network updates less frequently than the main network, which helps to reduce the variance in the updates to the Q-function.

Finally, the DQN algorithm uses a technique called experience replay, in which the agent stores a large number of past experiences (i.e., state, action, reward, and next state tuples) and samples from them randomly during learning. This helps by decorrelating experiences - learning from a diverse set of experiences that are not necessarily related in time, and stabilizing learning - by reducing the variance in the updates to the agent's policy especially in environments with long-term dependencies. 

Original [paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) (2015)
