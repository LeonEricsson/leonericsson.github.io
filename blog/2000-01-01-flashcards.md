---
layout: post
title: "flash cards"
categories: []
year: 2000
type: blog
---

flash cards 

---

- Generalization-centric paradigm: Data scale is relatively small. This paradigm further divides into two sub-paradigms:
  - Classical bias-variance trade-off (U-
    shaped) regime: Model capacity is in-
    tentionally constrained below the interpo-
    lation threshold (red dot • in Fig. 2). In
    this regime, both Generalization Gap and
    Approximation Error are non-negligible.
  -  Modern over-parameterized (second-
    descent) regime: Model scale signifi-
    cantly surpasses data scale (green dot •
    in Fig. 2). In this regime, Approximation
    Error is negligible.
- Scaling-centric paradigm: Large data and model scales, with data scale exceeding model scale (blue dot • in Fig. 2). In this regime, the Generalization Gap is negligible.
- The Bias-Variance Trade-off and the U-shape Regime. In the classic setting, the complexity of the training set T is typically smaller than the richness of the function class H = {fθ : θ ∈ Ω} and the absolute scales of the data and models are small. Conventional wisdom in machine learning suggests that one needs to carefully control the complexity of the function space H (Belkin et al., 2019) to balance Generalization Gap and Approximation Error:
  1. If H is too small (underfitting), all functions in H will have high bias, i.e. high Approximation Er-
    ror. This leads to a large training error, and thus a large test error.
  2. If H is too large (overfitting), the learned function may overfit the training data, leading to high
    variance. This results in a small training error but a large Generalization Gap (the difference be-
    tween test and training errors), and thus a large test error.
- Over-parameterization and the Second-descent Regime. The success of deep neural networks in tasks like image recognition around 2012 (Krizhevsky et al., 2012)
marked a (sub-)paradigm shift inside generalization-centric machine learning. Over-parameterized neu-
ral networks, possessing more parameters than required to perfectly fit the training data (interpolation
threshold), surprisingly continued to improve as they became even more over-parameterized, surpass-
ing the performance of under-parameterized models (He et al., 2016; Neyshabur et al., 2018). This phe-
nomenon, where increasing model complexity beyond the interpolation threshold leads to improved per-
formance, offered a new perspective beyond the classical bias-variance trade-off. It inspired the proposal
of the “double-descent curve” (Figure 3 (b)) to accommodate both the traditional U-shaped curve in
the under-parameterized regime and the observed single-descent curve in the over-parameterized regime
(Belkin et al., 2019).
- Heavy Under-parameterization and the Skydiving Regime. Breakthroughs in large language model pretraining leads us to a scaling-centric paradigm, distinguished
from the previous generalization-centric paradigm by two key features. First, the complexity of training
data T far surpasses the capacity of the models (Raffel et al., 2020; Brown et al., 2020; Hoffmann et al.,
2022; Touvron et al., 2023) and the training loss remains far from reaching a plateau. Second, both the
data and the models themselves operate at a scale vastly larger than in previous paradigms, as illustrated
in Figure 2 blue dot. Figure 4 (b) illustrates the typical learning dynamics in this paradigm: test and training error curves re-main closely aligned throughout training, even when model size and compute are scaled up by factors of
500 and 250,000, respectively. The training error has not yet reached its global minimum, suggesting fur-
ther scaling up either or both the model size and dataset size could lead to improved performance
- While L2 regularization is widely acknowl-
edged to reduce overfitting and thus enhance generalization
performance, our preliminary experiments indicate that it
may not offer similar benefits for language model pretrain-
ing. This aligns with current practices, as flagship language models such as GPT-3 (Brown et al., 2020),
PALM (Chowdhery et al., 2023), Chinchilla (Hoffmann et al., 2022), Llama-2 (Touvron et al., 2023) and
DeepSeek-V2 (DeepSeek-AI et al., 2024) do not employ L2 regularization. While weight decay is widely
used in training language models, our observations are consistent with Andriushchenko et al. (2023) in
that it does not play a conventional regularizer role.
- Conventional wisdom in neural network training often favors using a larger learning rate, possibly near
the maximum stable value, as this is believed to improve generalization performance (Li et al., 2019; Lee
et al., 2020; Lewkowycz et al., 2020). This practice is attributed to the implicit regularization of stochas-
tic gradient descent (SGD). While various learning rates can achieve perfect training accuracy for small
datasets, e.g. CIFAR-10, the gradient noise from larger learning rates is thought to guide SGD towards
better minima.
- Contrary to conventional wisdom, our findings reveal that the optimal learning rate for
large language models is significantly lower than the maximal stable value3 previously assumed. This sug-
gests that traditional regularization-based theories may not fully explain the dynamics of optimal learn-
ing rates in the context of training language models.
- In generalization-centric ML, a common observation is that, given the same computational budget (mea-
sured by the total number of training epochs), algorithms employing smaller batch sizes tend to generalize
better than those with larger batches beyond a critical threshold (Keskar et al., 2016; Smith & Le, 2017;
Shallue et al., 2019; McCandlish et al., 2018). This phenomenon is often attributed to the increased gradi-
ent noise associated with smaller batches, which acts as an implicit regularizer, mitigating overfitting.
