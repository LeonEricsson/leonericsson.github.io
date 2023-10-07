---
layout: post
title: "Training language models to follow instructions with human feedback"
categories: [NLP, Reinforcement Learning]
year: 2022
type: paper
author: Ouyang
exturl: https://arxiv.org/pdf/2203.02155.pdf
---
While next next token prediction objective has proven a remarkable tool for reasoning, comprehension, QA, completion emergence in large language models, it is fundamentally a different objective from "follow the user's instructions helpfully and safely" which one would like when interacting with it. If you've ever used some of OpenAI's earlier APIs you'll know how aggressive your prompt engineering has to be for a natural conversation to occur. This paper was fundamental for OpenAI alignment work, it's one of the most important techniques that made ChatGPT the phenomena it is today.

## Alignment
Alignment is an interesting topic because it's difficult to capture and define. Alignment as a concept is subjective, we as humans inherently disagree on a lot of things but despite this the goal of alignment research is fundamentally to align a model with "human values". Whose values? Well in the case of InstructGPT, the "who" is a combination of the human-labelers and their instructors. OpenAI chose to hire a team of 40 contractors to label their data, screened to be sensitive to the preferences of different demographic groups and harmful content. While "human values" is the overarching alignment objective, OpenAI decide to focus on three concrete values - helpfulness, harmfulness and truthfulness. Either way its important to remember that this paper demonstrates that their proposed alignment technique is successful in aligning a model to a human reference group for a specific application as opposed to all human values in any setting.

## Methodology
InstructGPT is developed in three concrete steps, contributing to several performance improvements. The family consists of three models at 1.3B, 6B and 175B derived from existing GPT-3 models. The process is illustrated in figure below.

The prompt dataset is a combination of prompts taken from 1) The OpenAI API 2) The human labelers. Most of the dataset contains prompts from the OpenAI API considering the amount of effort required for 40 human labelers to create meaningful and diverse prompts.  

### Step 1
A prompt is sampled from the prompt database and a human-labeler demonstrates the desired output behavior. This creates a supervised/labeled dataset of prompts $x$ and desired responses $y$ that is used to fine-tune GPT-3 base models and create supervised fine-tuning models (SFT). This process is straight-forward and in itself contributes with significant alignment. The SFT dataset consists of 13k data points.    

### Step 2
The SFT model is prompted and several outputs are saved creating a new dataset of prompt $x$ and $K$ outputs $y$. A human-labeler is asked to rank the responses internally from best to worst. In the original RLHF paper, only two model outputs are compared from the same input but for InstructGPT the labelers are presented with anywhere between 4 and 9 responses. This produces $K\choose 2$ comparisons for each prompt and you might be thinking "great, more data points for our model!" but unfortunately this overfits quickly. When each comparison is treated as a single data point then each completion will potential be used for $K - 1$ gradient updates. Instead what they did was bind the $K\choose 2$ comparisons into a single batch element which means each completion only requires a single forward pass.

After labeling, the dataset is used to train a reward model (RM) which is initialized from the SFT model with the final unembedding layer removed. RM's objective is simple - predict which of 2 completions the human labeler prefers. 

### Step 3
Now we've reached the reinforcement learning stage where PPO is applied to optimize a policy (SFT) against human preference, represented by the reward model. The environment can be thought of as a bandit environment; a prompt is sampled from the prompt database and fed to the policy network. The policy network generates a output (think response) and the reward model "rewards" the output following which the episode ends. The reward is used to update the policy network in traditional PPO fashion. 

## Evaluation
This process proves efficient in aligning models toward human performance as labelers (both held-out and normal) drastically prefer InstructGPT to GPT-3. Even when GPT-3 is prompted to be more instructional is InstructGPT preferred. On the test set, outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3. InstructGPT proves to be more truthful than GPT-3 equivalents. Public NLP datasets are not reflective of how actual language models are used, despite showing worse performance on benchmarks the model output is still heavily preferred by humans. 

## Cost of RLHF
GPT-3 was and still is a very large model, far beyond the capabilities of most research groups/companies. In the paper it is mentioned that GPT-3 took 3,640 petaflop/s-days to train. Assuming you have access to a (single) top-of-the-line tensor core GPU such as the A100 that equates to 11,666 days or 32 years. InstructGPT on the other hand "only" takes 60 petaflop/s-days (192 days on a A100) and the baseline SFT even less at only 4 petaflop/s-days. It becomes apparent that if we're trying to make LMs more helpful to us as humans, alignment in existing LMs seems more cost-effective than pretraining. That's pretty cool!