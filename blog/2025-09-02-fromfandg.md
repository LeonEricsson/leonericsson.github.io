---
layout: post
title: "From f(x) and g(x) to f(g(x))"
categories: []
year: 2025
type: paper
author: Yuan
exturl: https://husky-morocco-f72.notion.site/From-f-x-and-g-x-to-f-g-x-LLMs-Learn-New-Skills-in-RL-by-Composing-Old-Ones-2499aba4486f802c8108e76a12af3020
---

This post analyzes another entry in the ongoing debate about whether reinforcement learning (RL) teaches language models genuinely new skills. This topic has been a source of significant discussion, particularly since the release of a paper showing that `pass@k` on coding tasks did not improve during RL. This suggested that models weren't learning to solve problems they couldn't previously solve but were merely amplifying existing capabilities. This begs the question: do LLMs learn new skills during RL, and if so, what is learned and how do we incentivize it?

This blog argues that many of the previous negative results stem from inappropriate evaluation setups. The authors put forward a compelling counter-argument with what they call the **RL Compositionality Hypothesis**:

> *Once a model has acquired the necessary atomic, non-decomposable skills for a task through training, RL enables the composition of these skills into more complex capabilities when properly incentivized.*

-----

## Task Design

To test this hypothesis, the authors design a string transformation prediction task with several key properties. First, **atomic skills are well-defined**, allowing models to learn fundamental operations separately before RL. Second, **task difficulty can be controlled** by adjusting the compositional complexity of these atomic skills. Finally, the data is synthetic and designed to be **absent from pretraining corpora**, ensuring that any improvement is attributable to the training process, not memorization.

The core task involves a set of string functions (e.g., reverse, shuffle, repeat) with generic names like `func_16`. Given an input string $x$ and a composition of functions like $y = f(g(x))$, the model must perform deductive reasoning to predict the final output string.

-----

## Training

The training process unfolds in two stages. First, the model must acquire the **atomic ability** to predict the output of each single, non-composed function $f(\cdot)$. These skills are taught separately using reinforced fine-tuning (RFT), where the model is given the full function definition and a random input string.

Next, the experiment tests the model's ability to learn composition by comparing RFT against RL. Crucially, at this stage, no function definitions are provided to the model; the assumption is that they are now learned atomic skills. The difficulty is controlled by the nesting depth, where "Level x" refers to the compositional depth.

A **Level 1** problem involves a single function call:

```python
You are given a code:

def main_solution(x):
    return func_16(x)

Can you predict the output of `main_solution("tiheass")` without writing any code? Please reason and put your final answer in the following json format: {"output": <your output>}, where <your output> should be the final string.
```

A **Level 2** problem involves a nested function call:

```python
You are given a code:

def main_solution(x):
    return func_2(func_16(x), 3)

Can you predict the output of `main_solution("tiheass")` without writing any code? Please reason and put your final answer in the following json format: {"output": <your output>}, where <your output> should be the final string.
```

The authors compare two training algorithms on these compositional tasks:

  * **Composition via RFT**: This baseline mirrors the first stage but is applied to compositional problems (Level \> 1) with the function definitions hidden. This tests if the model can learn to compose simply by being trained on its own successful attempts.
  * **Composition via RL**: The model generates a response, receives a binary reward based on the correctness of the final answer, and is updated using GRPO (a PPO-style algorithm).

-----

## Experiments & Results

The first experiment trains the model using RL exclusively on Level 1 problems. Does the model spontaneously learn to compose these atomic skills? The answer is no. RL on atomic skills alone is insufficient for learning compositionality.

However, the picture changes dramatically once a small amount of Level 2 data is introduced into the RL training mix. Performance sharply increases on levels far beyond what the model saw in its training data. Seeing just a small sample of composition is enough for the model to generalize to much higher levels of complexity. In contrast, the same setup using RFT fails to produce this kind of generalization.

### Are Compositional Skills Transferable?

To push the hypothesis further, the authors test whether compositional skills are task-agnostic. They introduce a new task called **Countdown**, where the model must construct a mathematical expression from a given set of integers to reach a target number. The "level" of difficulty again corresponds to the number of components to reason about.

An example for **Level 3**:

```python
Using the numbers [95, 14, 18], create an equation that equals 99. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 * 4 </answer>.
```

The experiment compares four models on the Countdown task. The key comparison is between a model that has only learned the atomic skills for both tasks (via RFT) and a model that has also learned **compositional skills** on the string task (via RL).

According to the hypothesis, a model that has learned compositionality on Task A should be able to transfer this abstract skill to Task B, provided it has also learned the atomic skills for Task B. The results confirm this. The model trained with compositional RL on the string task significantly outperforms the others on the new Countdown task, demonstrating that the learned skill of "composition" is indeed transferable.

This finding reinforces the authors' hypothesis and offers valuable insights for practical model development. To improve performance across diverse tasks, it may be unnecessary to collect expensive RL data for each one. Instead, it's more efficient to equip models with a wide variety of atomic skills through cheaper methods like pre-training or SFT, and then incentivize the general skill of composition with RL on only a subset of tasks.

### A Second Look at `pass@k`

Finally, let's revisit `pass@k` on the string transformation task. At Level 1 and, to a lesser extent, Level 2, the results replicate the findings from the "reranking" literature—the performance gap between the base model and the RL-trained model closes as the number of samples ($k$) increases. This gives the illusion that RL is only amplifying what the model already knows.

However, this illusion fades as we test on harder compositional problems (Level 3+). For these more complex tasks, the RL-trained model maintains a significant performance advantage even at high values of $k$, demonstrating a clear acquisition of a new skill. This indicates that evaluation metrics like `pass@k` must be interpreted carefully. Without controlling for the underlying skill complexity and the model's pre-RL capabilities, such metrics can be misleading. In this experiment, where these variables are controlled, the data clearly demonstrates that RL can facilitate the acquisition of new, compositional skills that are absent in the base model.