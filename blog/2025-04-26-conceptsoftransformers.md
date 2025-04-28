---
layout: post
title: "next token prediction"
categories: []
year: 2024
type: blog
---
Mechanistic Interpretability is a tool to understand the internals of neural networks, specifically in this case transformer language models. Note that I'm saying **a** tool, not **the** tool. As any tool it has limitations, and in MI that usually involves looking at models through a microscope and trying to analyze certain, but limited parts of the behaviour in an attempt to create a broader picture of what is happening. It is important to understand that much work in MI is limited in this sense where behaviour can look present through studies but that doesn't account for every example tested. MI will often highlight the success cases where the authors actually managed to learn something interesting. The findings are often of the type of "existence proof" - concrete evidence that specific mechanisms operate in certain contexts. While you may suspect similar behavior in general this is where you cross the line of evidence into speculation. You typically have to examine a small subset of the entire model, either in a simplictic or reduced form to be able to interpret anything, but this always risks introducing bias.


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



**De-tokenization layers.** The early layers of a language model are involved in mapping the atrificial structure of tokens to a more natural semantically meaningful representation. Many early neurons respond to multi-token words or compound words. 

**Re-tokenization layers**. In reverse, the final layers of language models mediate conversion of words or contextualized tokens back into literal tokens of the vocabulary. There are neurons which convert or dictate a representation of the word into constituent tokens one by one for output. 

