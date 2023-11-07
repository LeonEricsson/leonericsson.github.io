---
layout: post
title: "Reasoning or Reciting? Exploring the Capabilities and Limitations of Language Models Through Counterfactual Tasks"
categories: [NLP]
year: 2023
type: paper
author: Wu
exturl: https://arxiv.org/pdf/2307.02477.pdf
---
Can LLMs genuinely reason, or are they merely regurgitating from their pre-training data? Researchers at MIT and Boston University spearheaded a study to find out. To decipher the reasoning capabilities of LLMs, they delved into their ability to generalize task knowledge, especially when subjected to problems of close proximity or similarity. To do so, they ingeniously crafted an evaluation framework based on 11 *counterfactual* task variants, which fundamentally diverged from the presuppositions of standard tasks.

## Counterfactual Tasks Explained

Counterfactual tasks, as introduced in the study, are an illuminating concept. They begin with typical tasks, like arithmetic, logic, or code generation, where LMs traditionally excel. However, by introducing minor tweaks or alterations to the foundational rules, the researchers created a new challenge. The core reasoning procedure remains intact, but the input-output mappings have changed. In layman's terms, if a human expert can solve a standard task, they should also be able to tackle its counterfactual counterpart given adequate time and the counterfactual's resemblance to the original task. Similarly, we'd anticipate LLMs, if they truly possess reasoning abilities, to demonstrate similar prowess.

However, it's crucial to clarify that the study wasn't geared towards constructing entirely novel or alien counterfactual models. For instance, Base-9 addition, in contrast to Base-10, isn't an innovative concept. Instead, counterfactuals are subtle variations of the primary task, where a reasoning-capable model shouldn't exhibit any significant drop in performance.

## The Revealing Results

The research team assessed a plethora of models, including GPT-4, GPT-3.5, Claude, and PaLM-2, across all tasks. Each model encountered both direct prompts and those nudging step-by-step reasoning, inspired by the chain-of-thought prompting mechanism.

![Model Performance](/public/images/reciteorreason.png)

GPT-4 emerged as the front-runner, outperforming its counterparts consistently. However, the overarching trend, evident from the above figure, is revealing. Across all tasks, LLMs struggled with counterfactual variants, often showcasing drastic performance dips. Such discrepancies insinuate that these models might be defaulting to rote memorization rather than genuine reasoning when faced with novel scenarios.

## Interpreting the Outcomes

* **Human Competence vs. Machine Limitations:** While humans might falter initially when introduced to an unfamiliar counterfactual task, they typically overcome the challenge given adequate time and reasoning. This adaptability underscores human competence in confronting new conditions, a trait currently lacking in LLMs.

* **Task-Specific Reasoning:** While LLMs' prowess in familiar tasks is commendable, the ideal model should encompass both specialized knowledge and overarching reasoning capabilities. This duality ensures that they can be leveraged in unprecedented situations, highlighting the difference between mere memorization and actual reasoning.

* **Significance of the Findings:** At first glance, the observed trends might seem reminiscent of the traditional train-test disparity in machine learning. However, an optimal learner should be endowed with the right inductive biases, allowing it to harness internal parameters and representations to craft general-purpose abstractions. These abstractions are pivotal in acing counterfactual tasks. The research underscores that today's top-tier LLMs haven't quite bridged this gap.

### **In Conclusion**

This comprehensive study offers invaluable insights into the inner workings of modern Language Models. While LLMs have undeniably revolutionized numerous sectors, it's evident that there's room for improvement, especially when differentiating between rote memorization and genuine reasoning. As the age-old adage goes, understanding our limitations is the first step towards overcoming them.
