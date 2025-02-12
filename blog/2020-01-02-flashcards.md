---
layout: post
title: "Flash Cards"
categories: []
year: 2024
type: blog
---

random, unordered thoughts / facts that are important and i want to remember

---

- In the generalization-centric paradigm, we operate on smaller scales compared to the LLM setting, enabling us to test ideas on datasets like CIFAR10 and scale them up to ImageNet if successful. Hyperparameter tuning at ImageNet scale is manageable, so we often disregard how hyperparameters should evolve with scale, assuming users will handle tuning. For example, introducing architectures (e.g., skip connections), data augmentation, or optimizers like AdamW doesn't require detailed scaling guidance for hyperparameters. In contrast, the "skydiving" regime faces challenges due to the immense scale of data and models, making traditional hyperparameter tuning impractical. Verifying new ideas or comparing approaches at this scale is prohibitively expensive. For instance, determining whether a global gradient clipping norm of 1 or 2 is better for a 100B parameter model demands immense resources, illustrating the difficulties inherent in this regime.
- Scaling-law crossover is a phenomenon where the effectiveness of techniques reverses at a critical scale—one idea outperforms another below this scale, but the opposite occurs above it. Unlike the traditional "test on CIFAR, scale to ImageNet" approach, evaluating ideas may require testing at progressively larger scales, demanding significant time, resources, and computational power.
- Years of research investment in understanding the training dynamics of neural networks has taught us
a valuable lesson: learning rates should be adjusted based on training specifics, particularly model scale
Goh (2017); Lee et al. (2019); Sohl-Dickstein et al. (2020); Xiao et al. (2019); Yang et al. (2022); Bi et al.
(2024). The challenge lies in determining how to adjust them effectively.
- Two learning rate proposals are compared: a constant learning rate, LR = 2/1024 (Proposal Blue), and a learning rate that scales with model dimension, LR = 2/D (Proposal Red). At small scales (D ≤ 1024), both approaches perform similarly, with Proposal Blue showing a slight advantage. However, at larger scales (D ≥ 1024), a crossover occurs, and Proposal Red consistently outperforms Proposal Blue, with the performance gap widening as scale increases. Additionally, scaling the learning rate with model dimension reduces training instability, even in the absence of QK-Norm.
-  The scaling-law crossover observed with Proposal Blue reveals that techniques like gradient normalization, GeGLU activations, and increased MLP dimensions, while effective at smaller scales, can lose their advantage at larger scales (e.g., 2–3 × \(10^2\) exaflops). This unpredictability complicates model optimization, raising questions about the reliability of innovations at extreme scales and requiring extensive resources to validate methods across progressively larger scales, such as \(10^4\) exaflops.
- Credit Assignment. The crossover scaling phenomenon underscores that demonstrating impressive performance at small scales is insufficient. While proposing new ideas and testing them at
small scales remains crucial, rigorously verifying ideas at large scales demands substantial effort,
resources, and, crucially, faith in their potential. Thus, we argue that scaling up existing ideas and
rigorously demonstrating their effectiveness at scale is as important as, or even more important
than, proposing new ideas and testing them on small scales. Both types of contributions are essential and should be recognized and valued
-  Avoid Biased Search Spaces. The scaling law crossover phenomenon indicates that it is crucial
to avoid overemphasizing ideas that work well at small scales. This narrow focus might lead us
to miss groundbreaking approaches, similar to the “vision transformer” (Dosovitskiy et al., 2020),
that excel at large scales but might not shine on smaller ones.
