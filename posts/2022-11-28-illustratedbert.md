---
layout: post
title: The Illustrated BERT, ELMo, co.
categories: [NLP, Transformer]
year: 2019
type: blog post
author: Jay Alammar
exturl: http://jalammar.github.io/illustrated-bert/
---

Blog post from Jay Alammar, mainly illustrating BERT. Unbeknownst to me, the classification part of the BERT architecture is simply the vector output from the CLS token. Great results were achieved using a classifier of just a single layer NN with the CLS vector as input. 

**Word Embedding Recap**

Word embedders, such as Word2Vec, transform words into numerical representations that capture semantic relationships (e.g. word similarity / “Sweden” and “Stockholm” have the same relationship as “Egypt” and “Cairo”) and syntactic relationships (e.g. “had” and “has” have the same relationship as “was” and “is”). Field quickly adopted these pre-trained embeddings instead of training them alongside their own model.

Word embeddings from Word2Vec will embed a word the same way no matter the context and naturally a question of context arose leading to contextualized word-embeddings which capture word meaning and context. ELMo was the most prominent contextual word embedder, it uses language modeling (predict next word in sequence) to learn the context using bidirectional LSTM. 

**Transformer revolution**

Transformers came in as a replacement for LSTMs partly because they could deal with long-term dependencies a lot better. The encoder-decoder structure made perfect sense in machine translation but how could it be used for sentence classification? How does one pre-train a language model that could be fine-tuned to downstream tasks? 

The OpenAI GPT transformer used a stack of decoders without the encoder-decoder attention sublayer used in the original transformer paper and strictly sticking to self-attention. You would simply feed it lots of books, naturally filled with long-term dependency examples, and let it predict the next word (language model) all while using the decoders masking property (no future peaking). Now all you had to do was tweak the fine tuning input slightly for each downstream task and you have an incredibly versatile language model capable of SOTA performance on a multitude of NLP tasks. But, what happened to bi-directionality?

BERT tackles this by using transformer encoders with randomly masked input tokens. The output vector of the masked token position is then used to predict the masked word. This adoption comes from an earlier concept called the Cloze task but in BERT it’s named “masked language model”. In addition to this masked prediction there is also a classification objective during pre-training where the model is asked to predict whether the two sentences fed into the network follow each other or not. This later improves the performance of certain downstream tasks such as question answering. 
