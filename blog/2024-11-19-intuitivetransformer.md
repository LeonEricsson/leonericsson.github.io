---
layout: post
title: "next token prediction"
categories: []
year: 2024
type: blog
---

we often talk about next-token prediction as the foundation of large language models. through this simple objective, the model gradually learns a broader understanding of the world. given a diverse dataset, this objective becomes incredibly complex, forcing the model to go beyond just the next word and instead grasp an entire context.

but despite its name, predicting the next token goes far beyond simply making a guess. i suspect there's some delusion about what actually happens. in truth, it’s about simultaneously predicting every possible token in the vocabulary—creating a relative weighting across that entire space. this distribution must produce the best token in expectation, and it encodes critical information about the transformer's state. some states are more deterministic—think programming or outputting json—so the entropy of the output distribution is lower. in contrast, creative writing demands a higher diversity of possible next states.

you should already see how learning to fit this distribution hinges on understanding and dissecting your context window. it’s more than just picking what’s most likely; you need to weigh the chances in a precise way, because some contexts call for considerable variety. that approach gives you the best chance of achieving a lower loss on average without overfitting to a single data point so heavily that the resulting distribution becomes skewed or unrealistic compared to what the data implies.

this distribution inevitably has to learn a structure that allows spontaneity and “learning from within the context,” because the data’s capacity far exceeds what you can memorize. you need to learn how to perform that internal learning.

of course, this distribution is trying to capture the underlying patterns of our training dataset, but we shouldn’t forget we have no direct knowledge of the “true” distribution. the model is trained as if the “correct” token is the only possible one, simply because we only ever see the actual next token in the data. it’s through scale that the model ultimately learns this incredibly rich distribution.

on average, you’re gonna get a lower cross-entropy loss if your distribution “hedges its bets” about what comes next. this hedging arises from fitting to the assumption that the next token is the most likely one—though it really isn’t. over time, you learn a distribution that captures variability across different contexts rather than always favoring a single token.

it’s precisely the tension between only ever seeing one actual token in training but needing to account for all those “latent” alternatives that pushes the model to generalize.

consider what happens when we dial the sampling temperature down to 0, effectively always sampling the most likely token. intuitively, that might sound fine—but in practice, models often get stuck in repetitive patterns, sometimes producing gibberish. why? because picking only the highest-probability token at each step magnifies even minor quirks in the learned distribution. you’re repeatedly committing to the model’s first choice, never giving it a chance to “escape” from a local pattern. if the model’s slightly biased toward a certain token or phrase, it can loop on that indefinitely. this is analogous to thermodynamic or simulated annealing approaches: maintaining some probability of deviating from the argmax helps avoid getting trapped in local minima.

in the generation process, each token the model outputs becomes part of the future context. if you deterministically pick the top token every time, any bias or loop-creating tendency in the model’s distribution gets reinforced at the next step. even a tiny preference can quickly spiral, leading to repetitive sequences. by sampling stochastically (with temperature in a sensible range), you give each step a non-zero chance to deviate, similar to how stochastic gradient methods can avoid local minima better than purely greedy descent.

