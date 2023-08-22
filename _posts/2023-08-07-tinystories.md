---
layout: post
title: "TinyStories: How Small Can Language Models Be and Still Speak
Coherent English?"
categories: [NLP, Transformers]
year: 2023
type: paper
author: Eldan
exturl: https://arxiv.org/pdf/2305.07759.pdf
---
We're back with another interesting paper published only a few months ago that was brought to my attention on twitter, sorry *X*. Still can't get over how dumb that change is. Anyways this paper reminds me of the Cramming paper where compute budget was restricted to one day but this time we're exploring how small datasets can create comprehensible models by curating the right data. Specifically the authors are hoping to train small language models (SLMs) capable of producing coherent, fluent and flowing texts. I think with all the capable models we've seen over the past few years, people have forgotten how "bad" models like GPT-2 (small), GPT-Neo, etc were att generating coherent text. They often produced illogical, repetitive and nonsensical answers which raises the question of whether the emergence of the ability to speak coherently requires excessive training, large models or complex architectures. This is the main motivation behind the authors research endeavour and is what lead to the creation of the TinyStories dataset.

## TinyStories
So, TinyStories as the name suggests, is a collection of short stories (3-5 paragraphs) that covers all the basic elements found in natural language intended to contain words that most 3-4 year olds would typically understand. The idea here is to capture the essence of language without all the breadth and diversity that is included in for example the Wikipedia dataset - that could potentially hinder a smaller models capability of just learning to speak coherently. To go about this, the authors use GPT-3.5 and GPT-4. But, just asking these LLMs for a story will not generate a sufficiently diverse dataset, even with temperature set to maximum, so for each generation the models are asked to incorporate three randomly selected words as well as a randomly selected feature. Something like this really speaks to the power of LLMs, given how little modification is necessary to be able to generate a dataset diverse enough to train new models, albeit SLMs. The authors don't mention if the final dataset is curated further to reduce replication or if they analyzed it in any way to determine its diversity. 

During evaluation of the models the authors continue using large language models by allowing GPT-4 to evaluate the models generated completions on grammar, creativity and consistency. They call this GPT-Eval. The actual evaluation dataset had been manually prepared by the authors and GPT-4 instead acts in place of the evaluation metric. Not too sure how I feel about using a probabilistic evaluation metric like this but the authors do mention that they average the score from 10 different completions by a model so I guess that's a good combatant. At the same time its pretty crazy that we're at the point where GPT can both create the training dataset and act as an evaluator of the models performance, wow. 

## Insight from GPT-Eval
This is more or less an copy of the list provided in section 3.1 of the paper but I thought it was clear and important enough to just write it down. These are some of the insights gained from training a bunch of different sized models on TinyStories and evaluating them at different time steps.
- Grammar is one of the first thing that emerges in LLMs and can be mastered by relatively small language models

- Consistency and creativity require larger models and doesn't plateau as early in training as grammar does.

- Generating a completion that is consistent with the beginning of the story emerges when the hidden size of the model increases from 64 to 128.

- The largest model trained on TinyStories (with 80M parameters) reaches almost perfect score on grammar and consistency but falls well behind GPT-4 on creativity alluding to the fact that creativity continues to improve substantially with model and dataset size. 

- When evaluating models of similar parameters size models with fewer layers show difficulty at staying in context which suggests that more layers are required to capture long-term dependencies.

- The authors hypothesize that knowledge of facts is more strongly related to embedding dimension whereas for context-tracking the number of layers is important. 