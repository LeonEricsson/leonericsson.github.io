---
layout: post
title: "language models. world models."
categories: []
year: 2023
type: blog
author: Gurnee
exturl: https://arxiv.org/pdf/2310.02207.pdf
---
The boring take is that Language Models memorize a massive collection of correlations without any fundamental model or *understanding* of the underlying data generating process given text-only training. The far more interesting alternative, is that in the process of compressing our data, they learn a compact, coherent and interpretable model of the generative process underlying the training data, i.e., a *world model*. 

In this paper, the authors explore the latter hypothesis, specifically through the question of whether LLMs form a world (and temporal) models as literally as possible - attempting to extract an actual map of the world. They argue that, while the existence of such a spatiotemporal representation does not constitute the dynamic causal world, having a coherent multi-scale representation of space and time is a precursor for a more comprehensive model.

So, how does one explore the existence of such a world map? Well, it turns out that we have to look at the hidden representation of our language models, turning to a method called **probing**. 

## Probing
The core idea of probing is to _use supervised models (probes) to determine what is latently encoded in the hidden representation of our target models._ This process entails analyzing the hidden representations at specific layers of the target model when it processes task-specific data. By fitting a small, linear model to these hidden representations, we can discern whether the representation contains information pertinent to a particular task of interest.

To conduct probing, one must:

1. Identify the specific hidden representation, $h$, you wish to examine, typically the activation (output) of a particular layer in your model.
2. Input a series of examples, $s_1, s_2, \ldots, s_N$, into your model and collect the corresponding hidden representations, pairing each with a task label that signifies the type of information you are probing for. 

| $X$   | $y$               |
| ----- | ----------------- |
| $h_1$ | $\text{task}_y^1$ |
| $h_2$ | $\text{task}_y^2$ |
| ...   | ...               |
| $h_N$ | $\text{task}_y^N$ |

3. Divide your dataset into training and testing sets, fit your probe model, and evaluate its performance.
4. A high accuracy of the probe model **indicates** that the probed property is encoded in the activations of the target model.

You are using your target model as a data engine, creating a dataset for which we can solve with a supervised learning approach in the form of a small linear probe model $\text{SmallLinearModel(X,y)}$. You can also see this as if you are fitting a model on top of your target models output, just instead of the typical final output you are looking at a internal representation of the output.

In this context, the target model serves as a data generator, creating a dataset that is then used in a supervised learning approach with a small linear probe model, \(\text{SmallLinearModel(X,y)}\). You can also see this as if you are fitting a model atop of your target model's output, but instead of using the final output, you analyze an internal representation.

It's important to note that probing can blur the lines with standard supervised model fitting. The information identified may reside within the probe model itself rather than the target model's data (activations). A more complex probe might reveal information not because it extracts more from the representations, but because it has a greater capacity to store information about the probed task.

Moreover, while probes can indicate what information the target model encodes latently, they do not confirm that the model utilizes this information for its intended behavior. Therefore, the findings from probing should not be construed as having a causal relationship with the target model's behavior.

## Probe to World Map
Back to the paper at hand, the authors construct a dataset constituting both space and time, through locations, news headlines and popular figures. These entities will be used on the target models (LLama 7B -> 70B) to generate the probe dataset. For a set of $n$ entities, it yiels a $n \times d_{model}$ activation dataset, for each transformer layer. The labels for this dataset consist of timestamps or two-dimensional latitude and longitude coordinates. The probe model used is linear ridge regression.

## Models of Space and Time
So, do models represent space and time at all? If so, where does this happen internally in our models? 

![](/images/spaceandtime.png)

I'm floored by how cool these results are. You can actively see the representations becoming stronger as you pass through the first half of the model! We clearly see how the larger models hold a higher capacity to represent space and time. These features are represented linearly within our massive model. 

This is far from definite proof that Language models have a complete internal representation of our world, it is especially hard to believe when on considers the apparent flaws in it's causal modelling, but its a sign of something forming in the latent space that our models exist in. and at the very least, it's exciting.

## World Model in Sora
Sora released a couple of days ago, again sparking a debate on the internal representations that exist within such models. I don't necessarily feel like dipping my toe in the discussion, but I found Yann LeCun's definition of world models particularly interesting so I'd like to share it with you.

Given:
- an observation x(t)
- a previous estimate of the state of the world s(t)
- an action proposal a(t)
- a latent variable proposal z(t)

A world model computes:
- representation: h(t) = Enc(x(t))
- prediction: s(t+1) = Pred( h(t), s(t), z(t), a(t) )
Where
- Enc() is an encoder (a trainable deterministic function, e.g. a neural net)
- Pred() is a hidden state predictor (also a trainable deterministic function).
- the latent variable z(t) represents the unknown information that would allow us to predict exactly what happens. It must be sampled from a distribution or or varied over a set. It parameterizes the set (or distribution) of plausible predictions.

The trick is to train the entire thing from observation triplets (x(t),a(t),x(t+1)) while preventing the Encoder from collapsing to a trivial solution on which it ignores the input.

Auto-regressive generative models (such as LLMs) are a simplified special case in which
1. the Encoder is the identity function: h(t) = x(t),
2. the state is a window of past inputs 
3. there is no action variable a(t)
4. x(t) is discrete
5. the Predictor computes a distribution over outcomes for x(t+1) and uses the latent z(t) to select one value from that distribution.