---
layout: post
title: "next token prediction"
categories: []
year: 2024
type: blog
---
Mechanistic Interpretability is a tool to understand the internals of neural networks, specifically in this case transformer language models. Note that I'm saying **a** tool, not **the** tool. As any tool it has limitations, and in MI that usually involves looking at models through a microscope and trying to analyze certain, but limited parts of the behaviour in an attempt to create a broader picture of what is happening. It is important to understand that much work in MI is limited in this sense where behaviour can look present through studies but that doesn't account for every example tested. MI will often highlight the success cases where the authors actually managed to learn something interesting. The findings are often of the type of "existence proof" - concrete evidence that specific mechanisms operate in certain contexts. While you may suspect similar behavior in general this is where you cross the line of evidence into speculation. You typically have to examine a small subset of the entire model, either in a simplictic or reduced form to be able to interpret anything, but this always risks introducing bias. 

## Framework for understanding the transformer
When I talk about the transformer in this blog I am referring to an autoregressive, decoder-only transformer language model. There are **a lot** of variants and modifications to this architecture, some more relevant than others, but the core blocks have remained the same for significant duration of time. It starts with a token embedding, followed by a series of layers or residual blocks as I prefer to call them, and finally a token unembedding layer. A residual block consists of an attention layer, followed by an MLP layer. The token (un)embedding layers transform a token, a single integer, into a *d*-dimensional float representation and vice versa.

To start, the embedding matrix transformers a sequence of $n$ tokens

$\mathbf{t} \;=\; (t_1,\,t_2,\,\dots,\,t_n) \;\in\; V^{\,n}$

into an embedding

$h \in \mathbb{R}^{\,n \times d}$ 

but for simplicity I want to focus on the final embedding in the sequence as it translates into the next token prediction. So we will focus on $h_n \in \mathbb{R}^{d}$ but I will omit the subnotation $n$. Let's have a look at this vector traveling through our transformer.

![alt text](image.png)

I really want you to focus on the residual stream here. Ignore exactly what's going on in the attention or MLP layer for now. Just focus on the flow of information. Notice how every layer in this architecture reads and writes to the residual stream. The residual stream doesn't do any processing itself, it acts as a communication channel for the layers. Every layer performs an arbitrary linear transformation to "read in" information from the residual stream and the start and performs another linear transformation before adding to "write" its output back into stream.

There are several very interesting effects of this linear structure.
The residual stream is a high-dimensional vector space. This means that layers can send different information to different layers by storing it in subspaces, without even interacting with intermediate layers. Once information has been added to a subspace it will persist unless actively deleted by another, this means that dimensions of the residual stream become something like a memory. 

Remember that we are only looking at the embedding at the single, final position in the sequence. And a natural question would be how information travels between positions in the sequence. This is where we get into the operations of the MLP and the Attention layers. Attention layers, or specifically attention heads, fundamentally act to move information between positions in the sequence. They read information from the residual stream of one token and write to the residual stream of another. Importantly, which token to move information from is completely separable from what information is "read" to be moved and how it is "written" to the destination. We divide the attention into two linear operations $A$ and $W_o W_v$, which operate on different dimensions and act independently.

- $A$ governs which token's information is moved from and to.
- $W_oW_v$ governs which information is read from the source token and how it is written to the destination token. 

The MLP layer operates independently on tokens.  

The token embeddings, as well as the unembeddings interact with a small fraction of the total number of dimensions. This means that most of our dimensions, and our vector is free to store information in. 

## Do Llamas work in English?
An mechanistic investigation into the latent language of multilingual transformers. Language models are primarily trained on English text, with a heavily biased distribution. Despite this they achieve strong performance on a broad range of downstream tasks, even in non-English languages. How do they achieve this generalization? This paper analyzes this through two distinct views. First, the premise, consider a translation task from french to chinese with the following few-shot prompt (where "中文" means "Chinese"):

```
Français: "neige" - 中文: "雪"
Français: "montagne" - 中文: "山"
Français: "fleur" - 中文: "
```

with the guarantee that the expected chinese translated word is a single-token word. This setup allows us to solely analyze the latent at position $n+1$, as it carries all the information necessary to produce the chinese translation.

The first method of analysis is a logit lens. A logit lens approach means that we take the models unembedding matrix $U$, which is normally applied after the final layer to get the logits which are turned into a next-token distribution, and instead apply after an intermediary layer $l$. This gives us an inspection tool to look at what tokens the model thinks is most probably as the latents travel through the model! This is very cool. Let us track the probabilities of the English translation word and the Chinese translation word throughout the layers, are we using english as a pivot language when performing this translation?

We find for the first half of the models layers, neither the chinese or the english analog garner any noticeable probability mass, then around the middle layer, english begins a sharp rise followed by a decline, while chinese slowly grows and, after a crossover with english, spikes on the last five layers. Wow, that is really cool, it does seem like english is used as a sort of pivot language. Looking at the entire next-token distribution throughout the layers, through entropy, we see that the first half layers is a uniform distribution with an entropy of ~14 bits (vocab is 32,000 so this makes sense) and then, at the same time as the english analog spikes in probability, the entropy sharply drops and remains low until the end of the model. 

Now, this is analysis provides a probabilistic perspective, studying the next-token distributions via the logit lens. The other promised perspective is that of a geometric one where we analyze the latens directly as points in Euclidean space. 

When you think of what a transformer is doing, you should be thinking of that of a vector moving through *d*-dimensional space. If we break it down. Following the embedding matrix, the starting position, before anything has happened in the model, is a latent / embedding 

$h $


## Anthropic: On the Biology of Large Language Models

#### Multilingual Circuits
Neural networks have highly abstract representations which often unify the same concept across multiple languages, this is well studied but how these interact provide us with a clearer understanding of the flow of a latent through the transformer. 

Consider three prompts with identical meaning in different languages:

- English: `The opposite of "small" is "` -> `big`
- French: `Le contraire de "petit" est "` -> `grand`
- Chinese: `"小"的反义词是"` -> `大`

analyzing the features, through what Anthropic calls *attribution graphs*, we find that there is significant overlap in the internal computation through overlapping "multilingual" pathways:

![](/images/multilingualpathways.png)
![](/public/images/multilingualpathways.png)

Notice how there is a language specific component separate from the shared multilingual components. The circuit works similarly across languages: the model recognizes, using a language-independent representation that it is being asked about antonyms of "small". This mediates a mapping from small to large, in parallel there is a feature track for the language which triggers the language-appropriate feature in order to make the final prediction. However, by inspecting the features we find indications that English is mechanistically privileged over other languages as the "default". The multilingual "say large" feature has stronger direct effects to "large" or "big" in English compared to other languages. We've seen this before (!) and it further established the notion that the models concept space is biased towards the english subspace likely as a result of biased pretraining.

The language pathway is interesting because it implies the existence of such multilingual features. To further prove it's existence, the authors identify a collection of antonym features in the middle layers of the model and similarly find synonym clusters in the same place. These features are identified through english prompts only. Then, prompting the models in using the same prompts as before, but this time negatively intervening on the antonym feature substituting in a synonym supernode, they find that the resulting output is a language-appropriate synonym! This demonstrates the language independence of the operation component in the circuit! The same procedure can be used on the prompt operand: `small` -> `hot` to derive language-appropriate antonyms of the word `hot`, and finally on the language itself, changing the output language to Chinese, French or English purely through modification of the language feature.


Taking the above to the extreme, one can look at large paragraphs of translated texts in a bunch of different languages, and look at the amount of overlapping features which activate at any point in the context. For each (paragraph, pair of languages, model layer), we compute the intersection (i.e the set of features which activate in both), divided by the total set of features which activate in either to measure the degree of overlap. If you do this for paragraphs which are direct translations of eachother and unrelated paragraphs in the same language pair you get a reference baseline for your comparison.

![](/images/featureoverlap.png)
![](/public/images/featureoverlap.png)

Looking at the beginning and end of models they tend to be language specific, consistent with the de/re-tokenization layer hypothesis.

**De-tokenization layers.** The early layers of a language model are involved in mapping the artificial structure of tokens to a more natural semantically meaningful representation. Many early neurons respond to multi-token words or compound words. 

**Re-tokenization layers**. In reverse, the final layers of language models mediate conversion of words or contextualized tokens back into literal tokens of the vocabulary. There are neurons which convert or dictate a representation of the word into constituent tokens one by one for output. 

This is why depth is important. This coincides with the token energy discussion seen before where the overall latent vector is mostly orthogonal to the token space throughout the early -> middle layers. While the features represented in the middle layers are multilingual they are mostly aligned with english as a result of biased pretraining. English is privileged in a way where the default output is English but the representation space is large, even though English is privileged doesn't mean models think in English.

