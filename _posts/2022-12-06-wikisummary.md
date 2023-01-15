---
layout: post
title: Generating Wikipedia by Summarizing Long Sequences
categories: [NLP, Transformer]
---
Original [paper](https://arxiv.org/pdf/1801.10198.pdf), Liu et al., 2018.

Researchers from Google Brain propose a variant of the original encoder-decoder Transformer containing only a stack of decoder modules which they suggest performs better on longer input sequences than RNNs (at the time still very popular) and encoder-decoder Transformers.

The paper considers the task of multi-document summarization where the input is comprised of a Wikipedia title and a collection of non-Wikipedia reference documents with a target being the actual Wikipedia text. They describe a first attempt to abstractively generate the *lead*. Here, the phrase *abstractively* is of importance, as it refers to the generation of new text as opposed to concatenating sentences from the input to form a summary (*extractive* generation).

As mentioned, the input material consists of cited sources from the wikipedia articles and web search results (top-10 cleaned for clones and the wikipedia article itself). Raw text input is created by a simple concatenation of paragraphs in order which is then encoded using sub-word tokenization with a vocabulary size of 32,000. Given very long input sequences (up to L = 11,000) the abstractive model, W, learns to write articles, treated as a sequence transduction problem.

Considering the traditional Transformer Encoder-Decoder's (T-ED) quadratic complexity in input sequence length the authors devised a simple yet effective modification that drops the encoder module (reducing model parameters by almost 50%), combines the input and output sequences into a single "sentence" and is trained as a standard language model. This is very similar to the approaches of BERT and GPT-2 that we've looked at earlier. This, combined with a few memory saving techniques, is what comprises the model architecture as proposed by the authors. This paper was one, if not the first, to propose splitting the traditional T-ED structure into models based solely on encoder or decoder stacks and it's since been used frequently in the language modelling domain.    


