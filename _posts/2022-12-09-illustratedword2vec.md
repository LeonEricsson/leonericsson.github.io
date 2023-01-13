---
layout: post
title: "The Illustrated Word2Vec"
categories: [NLP]
---
Word embeddings are central to the field of NLP. Word2Vec is one of the most influential word embedding algorithms and paved the ground for a lot of future research. Several concepts have transcended NLP and become effective in creating recommendation systems as well as the understanding of sequential data in non-language tasks.

### Word Embeddings

Taking a quick look at the general concept of word embeddings it's interesting to visualize some word vectors from GloVe. 

![](/images/gloveembed.png)

These are 50-dimensional word embeddings and we notice how some words are more similar than others, in accordance with our own understanding. For example, woman and girl are more similar than woman and boy but they are in turn more comparable than woman and water. Let's look at a famous example showing the power of word embeddings and their ability to capture word relations.

![](/images/gloveembed2.png)

Notice how the resulting "king-man+woman" vector is eerily similar to queen (although not exact).

### Model Training

Language models have a huge advantage over most other machine learning models as the amount of clean data is vast since we can train on running text. Word get their embeddings by looking at which words appear next to others. This is done by running a sliding window, of let's say 3 words, across a large corpora of text such as wikipedia. 

### Skipgram and Cbow

Remember the two architectures described the last time we looked at word2vec. Cbow generates a dataset similar to the sliding window process described previously but it introduces bidirectionality by trying to estimate a word in the middle of a sliding window. Skipgram follows a similar principle but instead of trying to guess the middle word it tries to guess all words around with the middle word as input. The ingenuity of this process is that from a sliding window of size 5 we get 4 samples instead of 1.

### Negative Sampling

Now that we have our dataset it would be natural to train our model as a next-word prediction task. This could easily be done with a neural network and standard SGD. The problem we have is that this is computationally expensive, especially in relation to the abundant data set we have and considering that this paper was published in 2013. So, instead of a NN approach, Word2Vec changes the next-word prediction task into a model that takes two words as input and predicts a neighbor score. This simple switch enables the use of a logistic regression model instead of a NN - allowing us to train on billions of samples. The only problem we have is that our dataset only consists of positive samples meaning our model is just going to return 1 every time. To address this, Word2Vec introduces *negative* samples. 

### Word2Vec Training Process

Now that we've introduced all the appropriate concepts we're ready to take a look at the true Word2Vec training process.

Training consists of two important matrices - the Embedding matrix and the Context matrix. Their dimensions are the same, vocabulary size * embedding size. In each training step a positive example is samples randomly along with its associated negative samples. We look up the input word in the Embedding matrix and the context/output words in the Context matrix. We generate output scores by calculating the dot product between the input word matrix and the context word matrices. These scores are turned into a probability distribution using a sigmoid and the ensuing error value is used to update the word embedding vectors. These embeddings continue to improve as we cycle through the dataset, following the training process the Context matrix is discarded and the embedding matrix used as our pre-trained embeddings. 



Original [paper](https://arxiv.org/pdf/1412.6980.pdf) (2014)
