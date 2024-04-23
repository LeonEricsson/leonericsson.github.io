---
layout: post
title: ""
categories: []
year: 2024
type: blog
---

People are starting to doubt LMSys after the Llama 3 release. This is unfounded. Llama 3 is a great model. LMSys just released it on the leaderboard too soon with high CIs. 
Sure, you can boost your scores somewhat om LMSys by creating a model that is nice to talk to but in the end that's what chatbots are for.

https://www.reddit.com/r/LocalLLaMA/comments/1c9nvpy/lmsys_becoming_less_useful/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-fineweb-15t-tokens-of-commoncrawl




There's been a discussion around scaling laws lately. There's a growing trend of training models way beyond the chinchilla compute optimal laws say. This is partly because
you trade training compute for later inference efficiency (e.g. train smaller models for longer) but as LLama3 noted in their release blog post "while the Chinchilla-optimal amount of training compute for an 8B parameter model corresponds to ~200B tokens, we found that model performance continues to improve even after the model is trained on two orders of magnitude more data. Both our 8B and 70B parameter models continued to improve log-linearly after we trained them on up to 15T tokens."

I think it's clear that the exact formulas proposed in the scaling papers don't apply. I think what people tend to forget is that those curves were established for the MassiveText dataset and particularly the irreducible term E is an estimation of entropy of natural text, that being the text in MassiveText.  





Phi-3 release seems like it's either benchmarkmaxxing or an insane outlier. Look at MMLU x Compute graph

https://twitter.com/natolambert/status/1782600141159174398

The paper was mostly nothing to read, the devil is in the data, as they say this themselves. They focus on optimal data regime as opposed to caring about the typical scaling laws, again training models FAR beyond the chinchilla optimal regime. Phi-3 doesn't address the architecture in any way but rather adopts previously successful versions from Llama2 and what they call "It follows the standard decoder architecture of a 7B model class, having 32 layers and a hidden size of 4096". The pre-training is divided into a phase-1 of mostly web sources followed by a phase-2 with synthetic and heavily filtered data. The authors note that instead of thinking of training models as being in a "compute optimal regime" or a "over-train regime" they focus on the quality of the data for a given scale. This means filtering the web to contain the correct level of "knowledge" and keep web pages that can improve "reasoning" capabilities. Anyway, is Phi-3 insane or not, we'll have to vibe check to see I guess. The benchmarks they post in the paper look crazy, but they've also got the numbers for Mistral, LLama3 and Mixtral way off??



This raises the year old question about how good our benchmarks actually are. Yes, we're back discussing this again. This was partly something that came up when reading the Phi-3 paper but also some threads on Twitter. I've never actually looked too deeply into the main benchmarks we use to gauge our models, now I think most people take benchmarks with a grain of salt but have you ever taken a look at the questions??

For example, MMLU is one of the primary sources of LLM judging. It's what most creators, authors, and spokespeople refer to when deciding on a SOTA model. Take a look at some of the Test set question on MMLU:

https://twitter.com/nearcyan/status/1782617805827031217

There are also a bunch of questions which are completely broken, these for example are complete test sampels drawn from the dataset

The complexity of the theory.?"1,2,3,4","1,3,4","1,2,3","1,2,4",C
Demand reduction,?"1,3,4","2,3,4","1,2,3","1,2,4",D
Predatory pricing.,?"1,2,4","1,2,3,4","1,2","1,4",D
The need to head off negative publicity.,?"1,3,4","2,3,4","1,2,3","1,2,3,4",C
They are too irrational and uncodified.,?"3,4","1,3","2,3","4,1",B
The purposes for which the information is used is in the public's interest.,?"1,2","1,3","2,3","1,2,3",A
How the code is enforced.,?"1,2,3","1,2,4","1,3,4","2,3,4",B

These questions lack complete context and are irrational. I'm conflicted about this, I read a thread from Greg Kamradt (creator of the Needle-in-a-Haystack test) that dismissed MMLU's value as a benchmark completely

https://twitter.com/GregKamradt/status/1781763505752072348

but I don't agree with this take: We want our models to reason and generalize, but at **the same time**, I want them to know stuff! So we need to benchmark both of this!



TruthfulQA suffers from exactly the same issue, what are these samples

https://twitter.com/nearcyan/status/1782625091156922482/photo/1

A lof of this seems like complete nonsense and it's not like this is the first time people are talking about this, I feel like discussions surrounding benchmarks have been going on for the past year yet we're still stuck using the same old ones. Here's a 7 month year old video about a GPT gaming MMLU

https://www.youtube.com/watch?v=hVade_8H8mE&embeds_referring_euri=https%3A%2F%2Ftwitter.com%2F&source_ve_path=Mjg2NjY&feature=emb_logo

When it comes to evaluating instruction-tuned models we've got a lot more sophisticated tools to do so imo. MT-Bench, Alpaca Eval 2, EQ-Bench are all great examples of this, I think everyone should check out AlpacaEval 2 since they released their length adjusted version their numbers are great. LMSys is continually working on improving Chatbot Arena and recently announced a new benchmark that's in the pipeline. I'm excited by all of this

https://arxiv.org/abs/2404.04475
https://twitter.com/lmsysorg/status/1782179997622649330

but on the base model side we're still grasping at straws.

