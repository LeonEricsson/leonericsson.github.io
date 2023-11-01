---
layout: post
title: "Dota 2 with Large Scale Deep Reinforcement Learning"
categories: [Reinforcement Learning]
year: 2019
type: paper
author: Berner
exturl: https://arxiv.org/pdf/1912.06680.pdf
---
Continuing on the game AI trend we have OpenAI Five, the worlds first Dota 2 AI that achieved superhuman performance after **10 months (!)** of training. Alongside other greats like AlphaGo and AlphaStar, OpenAI Five represents a remarkable stride in game AI that has captured both the imagination and attention of mainstream culture. I remember when this paper was released because I, as a former League of Legends devout, couldn't believe their claims, an machine beating world champions?! AI mastering Go, was more tangible; the game is fully observable, observations and actions are discrete, but actually beating world-champions in Dota seemed like worlds away. 

## Dota 2
To appreciate the complexity of what OpenAI achieved, let’s first delve into the mechanics of Dota 2:

- **Game Overview**: Dota 2 is a competitive MOBA where two teams, each comprising five players, strategically battle to defend their respective bases situated at opposite corners of a square map.
  
- **Game Dynamics**: Players control unique hero units, leveraging them to farm uncontrollable units on the map for gold. This gold is essential for purchasing items to enhance their heroes.

- **Complexity Factors**: 
  - **Long Time Horizon**: The game can last anywhere from 20 minutes to over an hour, requiring consistent strategic thinking.
  - **Partial Observability**: Not all events or units are visible, necessitating predictive game sense.
  - **High-dimensional Action and Observation Space**: The array of potential actions and observations in each game moment is vast.

## Training system
Humans interact with the Dota 2 game using a keyboard, mouse and computer monitor. They make decisions in real time, reason about long-term consequences of their actions, react to constant decisions from their opposing player, and more. How does one adopt a framework to translate the vague problem of "play this complex game at a superhuman level" into a detailed objective suitable for optimization? 

### Observation
The observation space for OpenAI Five is notably vast. At each timestep, a hero observes an astounding 16,000 inputs about the current game state. Rather than using screen pixels as input, OpenAI utilized arrays to approximate the in-game information available to a human player. While this approximation is imperfect, the model does receive **all** of this information simultaneously every time step, whereas a human would need to click/hover over various menus to retrieve that data. Below is a table of the full observation space, for each hero, per time step.

![](/public/images/dotaobservation.PNG)

*Key Data Normalization Strategy*: All float observations, inclusive of booleans, undergo z-score normalization. This means they are standardized by subtracting the mean and dividing by the standard deviation. Observations are clipped to a range of (-5, 5) for stability.

### Action space
Actions in Dota 2, for human players, involve combinations of mouse movements and keyboard commands. OpenAI Five abstracts these into a single *primary action* supported by several *parameter actions*. Primary actions span a wide range—from fundamental commands such as move and attack to more situational ones such as using spells or buying items. The actions available to a hero at each time step are determined through simple action filters and then presented to the model. 

Each primary action can have up to three associated parameters: 
1. **Delay**: Dictates when, within the next frameskip, the action occurs.
2. **Unit Selection**: Refers to one of the possible 189 units.
3. **Offset**: Represents a 2D coordinate for abilities that target specific map locations.

While the theoretical action space is enormous—approximately $1.839 \times 10^{6}$—in actual gameplay, the typical number of feasible actions oscillates between 8,000 and 80,000.

## Architecture
OpenAI Five's architecture is structured to process the complex, multi-array observation space efficiently. Observations are flattened into a singular vector, which then courses through a 4096-unit LSTM. This LSTM state subsequently projects the policy outputs, i.e., the possible actions and their associated value functions. 

![](/public/images/architecturesimpledota.png)

*Noteworthy Point*: While each of the five heroes on a team has a separate policy replica, most observations remain identical as Dota 2 promotes information sharing among team members.

### Observation processing
The input to the LSTM is a single observation vector that summarizes the state. Producing this vector is not as trivial as calling `.flatten()`. The observation space is tree-like, containing various data types that need to be combined into a single vector. OpenAI process the game state data-types according to the figure below. 

![](/public/images/flattenobsfive.png)

The game state vector is then combined with the controlled hero's Unit Embedding informing the LSTM which of the team's heroes it controls. 

### Action policy
The LSTM processes the observation vector and the output is projected to produce an action as shown here

![](/public/images/actionoutput.png)

## Reward shaping
Reward shaping remains a critical, albeit challenging, part of reinforcement learning. In an ideal scenario, a model would receive a singular reward at the game's conclusion: +1 for victory and -1 for defeat. However, given Dota 2's intricacy, a more nuanced reward system becomes imperative as we try to  simplify the [credit assignment problem](https://courses.csail.mit.edu/6.803/pdf/steps.pdf) and enable learning.

![](/public/images/rewardweightsopenfive.png)

## Training system
OpenAI Five's policy is trained using self-play PPO, quite reminiscent of AlphaZero. The difficulty here is not self-play, or PPO for that matter but creating a training, simulation, optimizing loop that is efficient. OpenAI achieve this through the following system:

![](/public/images/systemoverviewopenfive.png)

Rollout Worker machines run self-play Dota 2 games. They communicate in a tight loop with Forward Pass GPUs,
which sample actions from the policy given the current observation. Rollouts send their data to a central pool of optimizer GPUs which store it in local buffers called experience buffers. Optimizer GPUs, compute gradients using minibatches sampled randomly from its experience buffer. Gradients are averaged across the pool and used to update the parameters. The Optimizers publish the parameter versions to storage in the Controller, and the Forward Pass GPUs occasionally pull the latest parameter version.

## Ablation

### Batch size
The authors evaluate the benefits of increasing the batch size on small scale experiments and benchmark the performance gain against linear speedup. Ideally, using 2x as much compute should result in the same skill in 1/2 the time. In practice however, the authors find a less than ideal speedup, meaning sublinear. The speedup still exists, for example they find that a batch size of 983k hade a speedup factor of around 2.5x over the baseline batch (123k). They speculate that speedup may change later in training when the problem becomes more difficult and advocate for this type of scaling.

### Data quality
An unexpected feature of this optimization task is the length of the games; each rollout can take up to two hours to complete. Early in the project, the authors had rollout workers collect full episodes before sending it to the optimizers and downloading new parameters; similar to how one would expect a RL system to be designed. However, they found that this data was too old to be useful, often being useless or even destructive. In the final system, rollout workers and optimizers operate asynchronously: rollout workers download the latest parameters, play a small portion of the game (256 timesteps) and upload the data to the experience buffer with optimizers continually sampling from whatever data is present in said buffers.

*Staleness* is defined as the distance $M - N$ between the sample, generated by parameter version $N$, and the current parameter version (that we are optimizing for) $M$. An ablation study finds that increasing staleness causes significant training slowdowns. The final system design targets a staleness between 0 and 1 by sending game data every 30 seconds of gameplay and updating to fresh parameters approximately once a minute. The system that they've designed to make this loop work effectively is fascinating; I'm grateful that they've released such detailed studies behind its design.

*Sample reuse* is the ratio between the rate of optimizers consuming data and rollouts producing data. If optimizers are consuming samples twice as fast as rollouts are producing them, then on average each sample is being used twice and we say that the sample reuse is 2. This helps to understand the effect of optimizers sampling the same data from the experience buffer. They find that a sample reuse as little as 2 causes a factor two slowdown, and reusing 8 times prevents learning all together. The final design targets a sample reuse of 1.

