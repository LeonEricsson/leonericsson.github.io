---
layout: post
title: "Scaling Laws for Neural Language Models"
categories: [NLP, Transformer]
year: 2020
type: paper
author: Kaplan
exturl: https://arxiv.org/pdf/2001.08361.pdf
---
The authors study the scaling laws for language model performance for a variety of parameters. The conclusions from this paper aim to guide future training of fixed compute budget projects. The authors investigate to which extent architectural modifications affect the cross-entropy loss and in particular compare this to parameters such as model size, compute time and dataset size. Ultimately they aim to understand what is important when training LLMs. The setup and background of this paper is not of interest, the only thing to note is that the authors focus almost entirely on the Transformer architecture. I'll jump into presenting the summary and results provided by the authors as those are what gripped me the most. 

### Transformer shape
Through a series of investigations where the authors vary $n_{layer}$, $n_{heads}$ and $d_{ff}$ (dimension of feed-forward layer) while keeping the total non-embedding parameter count fixed the authors find that very little performance gains can be attributed to changes in these shape parameters. However, while the *source* of the non-embedding parameters doesn't seem to matter, the total number of these parameters does show a strong correlation with the test loss.

### Datasize and Compute
Similarly, model performance is strongly related to datasize (in number of tokens) and the computer budget. The authors even find that performance has a power-law relationship with each of the three scale factors N (non emedding parameters), D (dataset size) and C (compute) with no observed deviations of these trends. 

### Overfitting 
Performance improves as predicted when scaling both N and D in tandem, but these improvements are diminshed when one of the parameters is fixed. Interestingly the authors propose an exact scaling ratio to avoid such a diminishing penalty where the model size increased 8x only requires a 5x increase of the data. 

### Convergence is inefficient
If you are working with a fixed compute budget C but are unrestricted on both model size N and available data D, the optimal performance is achieved by training a very large model but stopping significantly short of convergence. I find this very interesting and in turn this means that large models are considerably more sample efficient than small models. Compute efficient training stops far short of convergence. 

### Final thoughts
The proposed scaling laws allow people to properly allocate scale parameters when training LLMs on a fixed compute budget. Despite the drastic increase in model parameters over the past years this paper emperically proves that performance is still highly dependend on scale as opposed to model shape. These results have most likely guided OpenAI in their future work which lead to a drastic increase in these scale parameters both in GPT-3 and GPT-4.


