---
layout: post
title: "The Unreasonable Effectiveness of Recurrent Neural Networks"
categories: [NLP, RNN]
year: 2015
---
Original [blog post](https://karpathy.github.io/2015/05/21/rnn-effectiveness/), Andrej Karpathy, 2015.

Regular neural networks are limited in their ability to only accept fixed-size vectors as input and produce fixed-size vectors as output. In comparison, the sequence regime enables more powerful models and enable us to build more intelligent systems. RNNs combine an input vector, with their internal (learned) state vector, essentially describing programs. *If training regular neural nets is optimization over functions, training recurrent neural networks is optimization over programs*.

# Recurrent Neural Networks

### RNN Computation
Jumping straight into it, the principles of a RNN are trivial. In the end, an RNN is just a step function called on some input x, resulting in an output y with a internal state h. Here, h, holds information of the entire history of inputs. Concretely, a RNN class could look like this:

```python
class RNN:
    def step(self, x):
        # update internal state
        self.h = np.tanh(x)
        # compute output
        y = np.dot(self.W_hy, self.h)
        return y
```

Similar to layers in regular neural networks, one layer isn't too interesting. But, if we start stacking RNNs where the output of one is used as the input to the next, we get something that works monotonically better.

# Character-Level Language Models
Given a long sequence of text, training a RNN character-level model is fairly straight forward and reminiscent of what we've talked about before with Transformer models. By encoding each character into a vector using one hot encoding and feeding them into the RNN one at a time we can observe the output probability distribution and train the internal neurons using backprop. At test time, feeding the RNN with a initial character and sampling from the output distribution we can have it ramble indefinitely.

# Teachable examples
It's fascinating to look at some of the examples provided by the author as they show us the capabilities and limitations of RNNs. Remember however that all of these are trained using relatively simple models. 

### Paul Graham generator
Concatenating the essays of paul graham we get data for a model that generates paul graham-like texts. The author trains a 2-layer LSTM with 512 hidden nodes on a 1 million character dataset. As an example we get the following passage: 

*The surprised in investors weren’t going to raise money. I’m not the company with the time there are all interesting quickly, don’t have to get off the same programmers. There’s a super-angel round fundraising, why do you can do. If you have a different physical investment are become in people who reduced in a startup with the way to argument the acquirer could see them just that you’re also the founders will part of users’ affords that and an alternation to the idea.*

This is very impressive considering this is just a character-level model with a fairly small dataset. It learns English from scratch and manages to spell words well with a few errors. It uses punctuation, apostrophes and spaces well. 

### Shakespeare
From a dataset of the works of Shakespeare we see that RNNs are able to adapt structure and style to their output. This is the result of a 3-layer RNN with 512 hidden nodes.

```
PANDARUS:
Alas, I think he shall be come approached and the day
When little srain would be attain'd into being never fed,
And who is but a chain and subjects of his death,
I should not sleep.

Second Senator:
They are away this miseries, produced upon my soul,
Breaking and strongly should be buried, when I perish
The earth and thoughts of many states.

DUKE VINCENTIO:
Well, your wit is in the care of side and that.

Second Lord:
They would be ruled after this chamber, and
my fair nues begun out of the fact, to be conveyed,
Whose noble souls I'll have the heart of the wars.

Clown:
Come, sir, I will make did behold your worship.

VIOLA:
I'll drink it.
```

Wow, the results are very impressive. This dataset was larger than the one previous, and it shows in spelling and punctuation but in addition to this it adopts the style and structure of Shakespeare's work.

I don't want to keep pasting in visual results from the original post, you can go view it yourself if your interested, but the authors continue to show examples of how effective the RNNs are. They try to push the structuring abilities further by giving it markdown, latex, and finally C code examples. The models are able to produce impressive results for all datasets. It's clear that long range dependencies are difficult for smaller models, for example one LaTeX output opened a proof environment and tried closing it as a lemma. The LaTeX results almost compiled, they just required minor tweaks. The C code didn't compile but the results are amazing nonetheless. The code is peppered with comments, it uses strings properly, pointer notation and handles brackets correctly. Again, most errors seem to depend on problems with long term dependencies. 

### Evolution of training

The author presents an interesting example of how the model evolves its training. First, it learns how to properly space words and end sentences. Then it slowly starts to produce common words such as "He", "His", "and" etc. Going forward it produces more and more correct words and after that it properly learns how to use most punctuation. When it's actually able to produce texts, which on a surface-level looks correct, it attempts to formulate proper sentences with long term dependencies, which is the most complex aspect of language modelling.  



