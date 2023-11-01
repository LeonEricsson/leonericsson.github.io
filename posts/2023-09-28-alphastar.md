---
layout: post
title: "Grandmaster level in StarCraft II using multi-agent reinforcement learning"
categories: [Reinforcement Learning]
year: 2019
type: paper
author: Vinyals
exturl: https://www.nature.com/articles/s41586-019-1724-z
---
DeepMind's foray into StarCraft II was ambitious. Beyond mere gameplay efficiency, AlphaStar assimilated both human and agent data, evolving strategies through deep neural networks. Designed with general-purpose learning methods, its architecture was meant to serve beyond StarCraft II.

## Understanding StarCraft II

### The Landscape of StarCraft II

In its core, StarCraft II is a 1v1 game where players select from three unique races—Terran, Protoss, and Zerg—each carrying distinct units, buildings, mechanics, and strategies. An added element of unpredictability is the Random race, providing a player with a race chosen at random. Victory is clinched when one player loses all their buildings.

## Supervised Learning: The Initial Step

Before delving into reinforcement learning, AlphaStar's journey began with supervised learning. This phase leveraged a dataset containing human game replays, serving multiple purposes:

1. **Feature Extraction**: The raw game data, though vast, contains a wealth of information on player strategies, common game progressions, and more. Extracting these features was crucial to provide an initial structure for the model.
2. **Behavioral Cloning**: By mimicking human strategies and game decisions, AlphaStar could generate a policy network that provides a good initialization for subsequent RL stages.
3. **Action Prediction**: Given a game state, predicting the next action or series of actions that a human might take was instrumental in understanding gameplay dynamics.

## Reinforcement Learning Approach

### Observation Space:

The observation space in StarCraft II is enormous. AlphaStar had to deal with:

- **Units & Buildings Status**: Health, type, position, and current action.
- **Resource Information**: Amount of minerals, vespene gas, and their rate of accumulation.
- **Visual Screen**: This includes the current camera view and the minimap. Elements like enemy visibility, terrain, and player-controlled regions had to be processed.

### Action Space:

Actions in StarCraft II aren't merely about moving a unit from point A to B. They encompass:

- **Macro Actions**: Base expansion, resource allocation, and tech tree decisions.
- **Micro Actions**: Unit micromanagement, skirmish tactics, and quick decision-making during battles.

### Rewards:

While the ultimate reward in StarCraft is victory, intermediate steps play a crucial role:

- **Tactical Rewards**: Successful enemy base scouting, defense against an adversary's attack, or securing a resource location.
- **Strategic Milestones**: Achieving tech upgrades, building advanced units, or gaining a territorial advantage.

### Policy Networks:

AlphaStar employed deep neural networks for its policy decisions. These networks, given the current state, would output a probability distribution over possible actions.

## Multi-Agent Reinforcement Learning System

DeepMind's approach hinged on a league of agents continually evolving and adapting:

1. **Self-Play**: The primary agent would play against its previous versions. This ensured that the agent didn't forget older strategies while learning new ones.
2. **League Play**: A league of agents with different strategies was created. As agents played against each other, they discovered and adapted to counter-strategies.
3. **Exploiter Agents**: Periodically, agents specifically designed to exploit weaknesses of the primary agent were introduced. This ensured that AlphaStar continually patched its vulnerabilities.
4. **Diverse Training**: By having agents with varying strategies, the training regime ensured a comprehensive exploration of the game's strategy space, preventing overfitting to specific strategies.


