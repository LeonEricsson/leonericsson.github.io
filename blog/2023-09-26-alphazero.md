---
layout: post
title: "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
categories: [Reinforcement Learning]
year: 2017
type: paper
author: Silver
exturl: https://arxiv.org/pdf/1712.01815.pdf
---
Looking back at an oldie but a goodie today, AlphaZero! The generalization of AlphaGo Zero that famously achieved superhuman performance in the game of Go. At the time, the study of games was already a long standing tradition in computer science. The solutions however, were generally ugly, muddled with domain knowledge priors and hand-crafted heuristics. AlphaZero on the other hand, is a completely generic algorithm without any additional domain knowledge except the rules of the game. It serves as a general-purpose reinforcement learning algorithm that ca achieve superhuman performance across many challenging domains, *tabula rasa*. In one of my earliest posts I covered AlphaTensor, a AlphaZero extension that discovered completely new matrix multiplication algorithms, proving that this general algorithm stood the test of time. If you are interested in a open source implementation of AlphaZero check out the [repo](https://github.com/LeonEricsson/AlphaFour) I built alongside writing this post.

## Representation
The board state, game rules and other information is represented using 3D image stacks that are fed into convolutional neural networks. It's transformed into a 3D image stack that serves as the neural network's input. This stack, denoted as $N \times N \times (MT + L)$, represents the game state by merging multiple sets of planes, each detailing board positions at specific time-steps. The $M$ feature planes are composed of binary planes indicating the presence of the player positions, with one plane for each piece type. This is represented for $T$ time-steps, giving the neural network temporal representation. There are an additional $L$ constant-valued input planes denoting the player’s colour, the total move count, and the state of special rules. 

## Policy and Value estimation
A neural network taking the above representation as input spits out a policy $\pi(a|s)$ and a value $v(s)\in [-1, 1]$. During self-play, the neural network is updated given the final outcome of the game as we train to minimize the error between the value estimator and the final outcome $z$. The underlying idea here is that the network will learn what states eventually lead to wins (or losses), as opposed to trying to manually value board positions and piece importance. Additionally, learning the correct policy gives a good estimate to what the best action is from a given state.  

## Self-play Training Pipeline
The aformentioned elements are brought together in a sophisticated iterative training pipeline involving four steps.

**Self-Play**: Starting with a neural network that has random weights (and thus a random policy), multiple games are played against itself. During each turn of a game, Monte Carlo Tree Search (MCTS) simulations are performed to select a move based on an "improved policy." This yields a training example with the current state, improved policy, and a yet-to-be-determined reward.

**Assign Rewards**: At the end of each game, rewards are assigned to each of the training examples generated during that game. The reward is +1 if the current player wins and -1 otherwise.

**Train Neural Network**: Once a set number of games are played, the neural network is trained on the collected supervised training examples to update the policy and value networks.

**Evaluation and Update**: The newly trained neural network is then evaluated against the previous version. If it wins more than a certain threshold of games, the new network replaces the old one. If not, another iteration is conducted to collect more training examples.

The pipeline iteratively refines the neural network to improve its ability to play the game by improving both value and policy estimation. 