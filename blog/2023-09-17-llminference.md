---
layout: post
title: "Inference, quantization, and the .cpp projects"
categories: [NLP, Transformer]
year: 2023
type: blog
---
Inference is a, if not the, hottest topic in open LLM research community right now. It seems like every day brings a fresh wave of innovative ideas to boost the performance of the expansive foundational models we have access to. While training models from the ground up remains a lofty challenge for many, the democratized nature of inference optimization means it's a popular for exploration by researchers of all levels. In this post, I'll delve into some recent reads, highlighting intriguing proposals and sharing my reflections on inference. While the format is a bit free-flowing, I hope you'll find a gem or two that sparks your interest!"

## Inference Optimization
This thing really blew up on my radar when I heard about the .cpp projects from [Georgi Gerganov](https://github.com/ggerganov). 
In the aftermath of Chinchilla, people have realized that we don't need models running hundreds of billions of parameters to actually generate good results and this is great because competent models like Llama can actually fit on consumer grade hardware, at least for inference! I mean even my M2 Air is capable of running inference on some of these models.

The unified memory setup on the new Apple Silicon laptops has opened up a space for consumer grade inference work that is attracting massive attention. Top-of the line consumer GPUs have 20+ GB of VRAM while my M2 Air (theoretically) has 16GB. The community support for projects such as [llama, sam, whisper].cpp is solid, and there is a huge interest in running OSS models on consumer hardware. There are constant optimizations at work here to reduce latency and optimize memory efficiency, it's all really cool to follow. For inference on regular GPUs, inference libraries such as HuggingFace Transformers or vLLM provide both easy installation and quick usage. The libraries come with dozens of internal optimizations that speed up inference beyond anything you'll reasonably be able to do on your own so check them out.

### Quantization
It is standard to train foundational models with FP16, or somewhere in that region at least. However, these models are generally too large to fit on consumer grade GPUs, unless you're running some distributed set-up. To counteract this, quantization of LLMs after training has gained popularity, a technique called GPTQ Quantization. Quantization is nothing fancier than reducing the number of bits needed for a weight and GPTQ does this by finding a compressed version of the weight that minimizes a mean squared error. These quantizations can save massive amounts of space. The llama.cpp project is focused on 4-bit quantization inference meaning that the models reduce up to 4x in size. 

| Model | Original size | Quantized size (4-bit) |
|------:|--------------:|-----------------------:|
|    7B |         13 GB |                 3.9 GB |
|   13B |         24 GB |                 7.8 GB |
|   30B |         60 GB |                19.5 GB |
|   65B |        120 GB |                38.5 GB |


Researchers however seem somewhat divided on the topic, for open-LLM enthusiasts who want to run inference at home, it's a given as the increase in feasible model sizes warrants the reduced precision but others seem skeptical of its performance degradation. At the time of writing this post, API costs are very promising given the amount of compute you get access to, it's hard to motivate running your own open-sourced LLM vs piggy backing of something like GPT-3.5 Turbo. [As I've mentioned earlier](/_posts/2023-08-23-llamagpu.md), LLM inference is notoriously memory-bound; companies like OpenAI can optimize GPU usage by stacking requests into large batches for the same cost as single input runs. As a toy experiment, let's take a look at the costs of GPT-3.5 Turbo and GPT-4:

| Model           | Context Size | Cost per 1K Tokens | Tokens per $1   |
|-----------------|--------------|--------------------|-----------------|
| GPT-4           | 8K context   | $0.06              | 16,667          |
| GPT-4           | 32K context  | $0.12              | 8,333           |
| GPT-3.5 Turbo   | 4K context   | $0.002             | 500,000         |
| GPT-3.5 Turbo   | 16K context  | $0.004             | 250,000         |

Assuming you can rent a solid GPU for about $2 dollars an hour you will need to be able to generate at least a couple of hundred tokens per second, constantly, for every hour you are paying for to match the prices OpenAI is offering. That is assuming you have a open-sourced model that is competitive with GPT-3.5 and GPT-4. You will most likely need to be able to handle a large batch size to achieve this meaning you need an solid VRAM and/or proper parallelism. It's possible, but it's hard to compete with the scale and utilization that OpenAI has already built. It is very likely that the larger companies are hosting these services at a loss to establish market dominance, similarly to how most tech startups have developed in recent times, but only time will tell how this pans out. I'm sure there are alternatives out there that I've missed, and there's no doubt this field will change drastically in the coming months. 


## Speculative Decoding
As we've already discussed, LLMs have this quirky behavior where processing a bunch of data almost takes as long as just a bit of it. This is mainly because a big chunk of their time goes into pulling data from memory.

The challenge, though, is that LLMs typically build sentences one word at a time, so how can we make use of parallel computing while still up-holding the sequential nature of text generation? Speculative decoding proposes we use a smaller model to draft a sentence first. Subsequently, this draft is batch-processed by the primary LLM. If the LLM's predictions align with the draft, it progresses to the next token. However, discrepancies between the draft and LLM's output necessitate discarding the draft and reverting to traditional processing. The efficiency of this method is rooted in the high likelihood of the draft model accurately predicting easy tokens, streamlining the overall process. The hard tokens where the big model disagrees "fall back" to original speed, at a small cost because of the extra draft model work. 

## Medusa
While the draft model is a smart solution to speed up inference and works surprisingly well given a ideal draft model, the solution is inherently complex which has limited wider adoption. Medusa is an alternative framework that was recently proposed which I happened to stumble upon during this inference tangent I'm on. There's a great figure providing an overview of the framework below.

![](/images/medusa.png)

It's a remarkably simple solution to a problem that is otherwise riddled with small complex optimizations. The medusa heads are akin to the language model head in the original architecture (the last feed-forward layer mapping hidden state to vocab), but instead of predicting the next token they predict multiple forthcoming ones. This is quite ingenious, after all transformer blocks we arrive with a final hidden state that typically is used to predict just the next token but instead they insert multiple heads in parallel used to predict further ahead. Now, you might be thinking what good this does us if we need to re-train the model with the medusa heads inserted? Well we don't! All we need is to fine-tune the model on the original corpus (or a new one generated by the model itself) with the original model frozen; only fine-tuning the medusa heads. The authors claim this converges extremely fast and is a highly parameter-efficient process. On the Vicuna models that were tested, Medusa heads achieve a top-1 accuracy of 60%, jumping to 80%(!) for top-5. 

While 60% is decent, and probably usable on its own, there's great interest in leveraging the massive jump to 80% accuracy for top-5 predictions. This leads us into the next part of Medusa; tree attention. The authors allow each Medusa head to produce a number of predictions and form set of candidates from the cartesian product of these predictions. These candidates are attented do using tree attention with an appropriate mask that restricts attention to token predecessors. All in all I think the methods here are very crafty, and the results indicate a 2-3x speedup in inference which is great. The medusa heads seem to be trainable on just a single GPU, in less than a day (depending on model size). 

## Current state of Local LLMs 
Local LLMs are extremely important for a lot of different actors in the field. OpenAI changes their API constantly, phasing out models, make changes to their TOS etc. Running models completely off-line, with a guarantee of data remaining in-house is in the interest of all companies looking to use generative AI. Unfortunately, no matter how many posts there are saying X model beats GPT on Y benchmark. The fact of the matter is that benchmark tests are **not** representative of real world performance. Some models may come close to GPT-3.5 on narrow tasks but we don't have anything close to GPT-4 at the moment. Models in the <30B parameter region typically require a lot of work, they have a hard time following instructions over time and your prompts have to be carefully crafted. We still lack a metric that translates well into every day use and we're still looking for something that comes close to the magic in GPT-4.