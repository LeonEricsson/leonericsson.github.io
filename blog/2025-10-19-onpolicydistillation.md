---
layout: post
title: "On Policy Distillation"
categories: []
year: 2023
type: paper
author: Agarwal
exturl: https://arxiv.org/abs/2306.13649
---
A bit of a mess in here, I started reading the on-policy distillation paper and went on a tangent reading about MLE and the connections to KL Divergence. Would not recommend reading this, too scattered.

## Maximum Likelihood Estimation
Imagine we observe samples drawn from a distribution $x \sim X$ where each observation is drawn independently from the domain with the same probability distribution (I.I.D). Density estimation describes the process of selecting a probability distribution function and the parameters of that distribution that best explain the joint probability distribution of the observed data.   How do we choose such function, and its parameters? The problem is made more difficult as samples drawn from the population are typically noisy. There are many techniques to solve this problem, although the two common approaches are Maximum a Posteriori (MAP), a bayesian method, and Maximum Likelihood Estimation (MLE), a frequentist method.

The MLE approach treats the problem as an optimization or search problem, where we seek a set of parameters that result in the best fit for the joint probability of the data samples. Let's parametrize our probability function and parameters, we define $\theta$ to be the choice of both. In MLE, we wish  to maximize the probability of observing $x$ from our probability distribution. Said probability is formalized as

$$
P(X; \theta)
$$

This conditional probability is referred to as the likelihood of observing the data given the model parameters. As mentioned, we turn this into an optimization problem

$$
\max P(X; \theta)
$$

Remember that X consists of several observed data samples (x1, x2, x3, ... xn). This means $P(X; \theta)$ can be restated as

$$
\prod_{i=1}^n P(x_i; \theta)
$$

because of the I.I.D assumption from before. Two independent events A, B: P(A,B) = P(A) * P(B) which extends to n independent observations $x_i$. Multiplying many small probabilities together can be numerically unstable, it is therefor common to restate this as a sum of log probs.

$$
\sum_{i=1}^n \log(P(x_i; \theta))
$$

Given the frequent use of log in the likelihood function it is commonly referred to as a log-likelihood function. In optimization problems we prefer to minimize the cost function rather than to maximize it. Therefore, the negative of the log-likelihood function is used, referred to as a Negative Log-Likelihood (NLL) function.

$$
\min (-\sum_{i=1}^n \log(P(x_i; \theta)))
$$

#### Relationship to supervised learning

  

This provides a probabilistic framework for predictive modeling that is easy to extend into common machine learning problems such as supervised learning. In supervised learning we have pairs of data (x,y), slightly different from the case we've discussed so far. However, we can readily generalize the above formulation to the case where the goal is to estimate a conditional probability in order to predict y given x:

$$
\max (\sum_{i=1}^n \log(P(y_i| x_i; \theta)))
$$

Which means that the same MLE framework used for density estimation can be used to find a supervised learning model and parameters. This provides the basis for foundational linear modeling such as linear regression and logistic regression.  This formulation is the basis for **many** supervised loss functions that you are probably well familiar with.  Note that the above formulation defines the MLE objective, but in machine learning we want to minimize a loss function which means we turn it into NLL for the loss function. Let's make this concrete by looking at binary classification and binary cross entropy loss. **You'll see that cross-entropy loss *is* the Negative Log-Likelihood for a classification model that assumes a Bernoulli probability distribution.**  

**Binary Cross Entropy**
The first step is to define our probability distribution $P(y_i| x_i; \theta)$. The definition depends on the task at hand, in this case we have a supervised learning problem with a **classification task**. In binary classification, the true label $y$ is either 0 or 1. This is a perfect match for the Bernoulli distribution which models a single trial with two outcomes. Imagine now that our model outputs a single value $\hat y$ which represents the *probability* that the true label is 1:

- $P(y_i| x_i; \theta) = \hat y$
- $P(y_i| x_i; \theta) = 1 - \hat y$
  
This can be rewritten into a clever equation called the Bernoulli probability mass function

$$
P(y_i| x_i; \theta) = \hat y_i^{y_i} \times (1 - \hat y)^{1 - y_i}
$$

With this formulation of our underlying probability distribution, that is by making an assumption that the underlying data is distributed according to a Bernoulli distribution, we have a formula that we can now directly optimize for.  Let's move forward by applying the MLE framework from above on this probability. Our goal is to find the parameters $\theta$ that maximize the log-likelihood of this probability.

$$
\log (P(y_i| x_i; \theta)) = \log (\hat y_i^{y_i} \times (1 - \hat y)^{1 - y_i}) = {y_i}\log (\hat y_i) + (1 - y_i)\log(1 - \hat y)
$$

This is the objective. Maximizing a value is identical to minimizing its negative, hence our loss function is:

$$
\text{Loss} = -[{y_i}\log (\hat y_i) + (1 - y_i)\log(1 - \hat y)]
$$

This expression is the definition of **Binary Cross-Entropy Loss**!

**Cross Entropy**
The exact same principle can be applied to derive the general cross-entropy loss. Now we swap the Bernoulli distribution to the more general multi-class Categorical distribution which describes the possible results of a random variable that can take on one of $C$ possible classes. First we define the probability $P(y_i| x_i; \theta)$. Assume our model outputs a vector $\hat y_i$ of $C$ probabilities that sum to 1 (using a softmax function for example). The true label $y_i$ is a one-hot encoded vector. The probablity mass function $P(y_i| x_i; \theta)$ can be written as

$$
P(y_i| x_i; \theta) = \prod_{c=1}^C \hat y_{ic}^{y_{ic}}
$$

---

A small ingress here before we continue, you may be asking why we start with the PMF. Where does it come from?  Remember, our MLE objective hinges on calculating the likelihood of our observed data. For a single sample, that is $P(y_i| x_i; \theta)$, which reads: "What is the probability of observing the specific class we saw $y_i$, given the input $x_i$ and our model parameters $\theta$?" For this we must ask what kind of data is $y_i$?

- **Classification (Discrete):** The label $y_i$ is a specific category, a one hot encoding. The set of possible outcomes is finite and distinct
- **Regression (Continous):** The label $y_i$ is a measurement. It could be 1.23, 1.230001, ... the set of outcomes is inifnite

This distinction matters because to find the probability of a discrete outcome you use a probability mass function. We start from the PMF in classification because it is, by definition, the correct mathematical tool for answering our question: "What is the probability of _this exact, discrete_ outcome $y_i$?" The assumption we make here is that our data $y_i$ is drawn from a Categorical distribution. Our model (neural network + softmax) does not define the PMF, our *statistical assumption* does. The job of our model is to provide the parameters for the PMF. The output from our model $\hat y_i$ is the parameters of our PMF for one sample which we retrieve by feeding $x_i$ into our model $f(x_i; \theta)$. In short: We need a definition for $P(y_i| x_i; \theta)$. Because yi is discrete, that definition _must_ come from a PMF. The choice of which PMF (Categorical, Bernoulli, Poisson, etc.) is our modeling assumption, and our neural net's job is to compute the parameters for that PMF.

---

Let's move on. Apply the MLE framework: find the $\theta$ that maximizes the log of this probability.

$$
\log (\prod_{c=1}^C  \hat y_{ic}^{y_{ic}}) = \sum_{c=1}^C \log ( \hat y_{ic}^{y_{ic}}) = \sum_{c=1}^C y_{ic}\log \hat y_{ic}
$$

Note: the PMF formulation, and by extension this objective is just the log-probability of the single correct class: $\log \hat y_{ic}$, because $y_{ic} = 0$ for all incorrect classes and $=1$ for the correct class (one-hot vector). The MLE objective is to maximize this value, that is to maximize the log probability, or just probability of the correct class.

The loss function is again just the negative log-likelihood

$$
\text{Loss} = -\sum_{c=1}^C y_{ic}\log \hat y_{ic}
$$

This is coined the **Cross-Entropy Loss**.

Why cross-entropy, the name does not seem connected to what we've derived so far. Cross-entropy is a measure from the field of information theory, building upon entropy and generally calculating the difference between two probability distributions. It is closely related to but is a different from KL divergence that calculated the relative entropy between two distributions. For a discrete probability distribution cross-entropy is:

$$
H(p, q) = - \sum p(x) \log q(x)
$$

Exactly the same as our derived negative log-likelihood from before. The two concepts, one from statistics and one from information theory converge on the exact same mathematical formula.

There even broader connections we can make between these two fields if we take a step back. Taken from page 132 of Deep Learning book.  Let  $p(x)$ be the true data distribution which is unknown and $q(x; \theta)$  be our modeled distribution which we are trying to learn. We are trying to achieve this by finding good estimates to its parameters $\theta$ . This can be seen as minimizing the KL divergence between them

$$
\min_\theta D_{KL}(p|| q) = \min_\theta \mathbb{E}_{x \sim p} [\log p(x) - \log q(x;\theta)] = \text{p(x) does not depend on model parameters} = \min_\theta \mathbb{E}_{x \sim p} [-\log q(x;\theta)] = \max_\theta \mathbb{E}_{x \sim p} [\log q(x;\theta)]
$$

Now, remember the general MLE objective from earlier

$$
\max \sum_{i=1}^n \log P(x_i; \theta)
$$

We can turn this into an expectation (by dividing by $n$) because rescaling the cost function by does not effect the argmax:

$$
\max \mathbb{E}_{x \sim X} [\log P(x_i; \theta)]
$$

So. What does that mean? Minimizing the KL divergence ==  Maximizing Likelihood == Minimizing Negative Log-likelihood. Previously we saw that the formula for cross-entropy for a discrete distribution was the same as our derived negative log-likelihood for a categorical distribution. But this does not capture the full picture. Cross entropy between two distributions q and p is defined as:

$$
H(p, q) = - \mathbb{E}_{x \sim p} [\log q(x)].  
$$

While the entropy of p is:

$$
H(p) = - \mathbb{E}_{x \sim p} [\log p(x)].  
$$

and the definition of KL being:

$$
D_{\mathrm{KL}}(p | q) = H(p, q) - H(p),  
$$

In our problem setting, $H(p)$ is independent of $q$ (and model parameters $\theta$), so minimizing $D_{\mathrm{KL}}$ w.r.t $\theta$ is equivalent to minimizing $H(p, q)$, i.e minimizing the cross-entropy between $p$ and $q$.  Quote directly from the book: "*Many authors use the term 'cross-entropy' to identify specifically the negative log-likelihood of a Bernoulli or Softmax distribution, but that is a misnomer. Any loss consisting of a negative log-likelihood is a cross-entropy between the empirical distribution defined by the [true distribution] and the probability distribution defined by the model.*"  From a statistical approach, we seek a set of parameters that result in the best fit for the joint probability of the observed data samples, where we turn this into a optimization problem my defining our MLE objective

$$
\max_\theta \sum_{i=1}^n \log(P(x_i; \theta))
$$

where $\theta$ are model parameters. In this problem setting, where we are seeking for max w.r.t $\theta$, this is equivalent to minimizing the KL divergence between the true (underlying) distribution that $x_i$ is sampled from, and the probability distribution of our model. This is in turn equivalent to minimizing the cross-entropy loss.

Typically, when performing Maximum Likelihood Estimation for a model that assumes a Categorical probability distribution, we call the negative log-likelihood of that the Cross Entropy Loss. This is a misnomer, because any time we are minimizing NLL we are minimizing the cross-entropy between the empirical distribution and the model distribution, this is not exclusive to Categorical distributions. **We can thus see maximum likelihood as an attempt to make the model distribution match the empirical distribution**. Ideally we would like to match the true underlying distribution but we do not have access to it. The optimal $\theta$ is the same regardless if we are maximizing likelihood or minimizing KL divergence.

##### Regression
Until now we've assumed the true distribution which we are trying to model is discrete, now let's look at  the continuous case. Let's remember that the maximum likelihood estimation framework gives us a consistent framework for predictive modelling which we can use to create optimization problems for our task. The appeal of MLE is that it can be shown to be the best estimator asymptotically, as the number of samples $n -> \infty$.

Anyway, for the regression case we first define our probabilistic model. Now  we assume that for a given $x_i$, the corresponding continuous label $y_i$ is drawn from some continuous distribution, commonly the Gaussian distribution. Our goal is to find the maximum likelihood distribution. Our model, neural network, does not directly produce a Gaussian distribution, instead we design our model $f(x_i; \theta) to predict the mean of this Gaussian distribution. Our prediction is $\mu = \hat y_i = f(x_i;\theta)$. We also assume that the noise is the same for all samples, that is a constant variance $\sigma²$. Because $y_i$ is continuous we use the probability density function for the Gaussian distribution to find its likelihood:

$$P(y_i | x_i; \theta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(y_i - \hat y_i)^2}{2\sigma^2} \right)$$

where we've swapped $\mu$ for our prediction $\hat y_i$. As always, our goal is to find the parameters which maximize the sum of the log likelihoods for all $n$ data. With some derivation this becomes:
$$\max_{\theta} \sum_{i=1}^n \left[ \log\left(\frac{1}{\sqrt{2\pi\sigma^2}}\right) - \frac{1}{2\sigma^2} (y_i - \hat{y}_i)^2 \right]$$

We can drop the log term because it does not depend on our model parameters. The same reasoning can be applied to $\frac{1}{2\sigma^2}$ as it just scales the loss function and won't change the location of optimal $\theta$. We swap to min to remove the negative factor and end up with:
$$\min_{\theta} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$
This expression is the sum of squared errors. Minimizing this is equivalent to minimizing the Mean Squared Error. You may have heard about using MSE before in regression and not questioned where it came from. This provides a formal justification for it, as it is equivalent to maximizing the log-likelihood. Further, based on our previous connections between NLL and cross-entropy, we can say that minimizing mean squared error w.r.t to theta is the same as reducing the cross-entropy between the empirical distribution and a Gaussian model.  

---

## Paper

**Knowledge Distillation** is a well established method for compressing certain skills from a trainer into a student. To be able to properly serve a wide variety of use cases, cost, speed, quality, being able to compress knowledge into smaller version of a model is very useful and has been used in many cases to improve smaller models in a model family release. Distillation is a great way to transfer certain tasks effectively into a student model, assuming that the model has capacity for the task. This, as we will see, can be more effective than trying to learn the task yourself.

**Forward KL is maximum likelihood**
"Forward KL under an empirical data distribution corresponds to maximum likelihood, which we optimize in supervised learning"

Forward KL is
$$

D_{\mathrm{KL}}(p | q) = \mathbb{E}_{x \sim P} \log P(x)/Q(x) = \mathbb{E}_{x \sim P} \log P(x) - \log Q(x)

$$
Maximum Likelihood is
$$

\max_\theta \sum_{i=1}^n \log Q(x_i; \theta)

$$
with $x_i$ from an empirical distribution. This can be turned into an expectation because rescaling does not effect the max
$$

\max_\theta \mathbb{E}_{x \sim P} [\log Q(x_i; \theta)]

$$
If we are minimizing KL w.r.t some model parameters, then P(x) does not depend on these parameters so we can ignore that term and write:
$$

\min_\theta D_{\mathrm{KL}}(p | q) = \mathbb{E}_{x \sim P} - \log Q(x; \theta)

$$
which is equivalent to Maximum Likelihood. So saying they are always equivalent is not correct I think, but in this case, if we assume a learning problem it is correct to say that minimizing forward KL is the same as maximum likelihood.

### Distillation

The paper's starting point is the standard **Supervised Fine-Tuning (SFT)**, which we've established is a conditional **Maximum Likelihood Estimation (MLE)** problem. The objective $L_{SFT}(\theta)=\mathbb{E}_{(x,y)\sim(X,Y)}[-log~p_{S}^{\theta}(y|x)]$ seeks to find student parameters $\theta$ that maximize the log-probability of ground-truth sequences. As we've derived, this MLE objective is mathematically equivalent to minimizing the **forward KL divergence** $D_{\mathrm{KL}}(\hat{p}_{\text{data}} || p_S^\theta)$, where $\hat{p}_{\text{data}}$ is the empirical distribution represented by "hard" one-hot targets. Given that the entropy of this deterministic (one-hot) distribution is zero, this optimization is identical to minimizing the **cross-entropy** $H(\hat{p}_{\text{data}}, p_S^\theta)$. In practice, this simplifies to minimizing the negative log-probability of the single correct token at each step in the sequence.

The classic **Supervised Knowledge Distillation (KD)** modifies this by changing the target distribution. Instead of learning from "hard" one-hot targets, the student learns from the "soft" full probability distribution provided by the teacher, $p_T$. The objective is stated as minimizing the forward KL divergence: $L_{SD}(\theta):=\mathbb{E}_{(x,y)\sim(X,Y)}[\mathcal{D}_{KL}(p_{T}||p_{S}^{\theta})(y|x)]$. This is a "distribution matching" problem where the student is trained to mimic the teacher's "thought process."

This KL minimization is equivalent to minimizing the cross-entropy $H(p_T, p_S^\theta)$. The teacher's own entropy, $H(p_T)$, is a non-zero constant, but as it does not depend on the student's parameters $\theta$, it drops out of the optimization gradient. The resulting loss, $-\sum_{c} p_T(c) \log(p_S^\theta(c))$, provides a much richer training signal than the SFT objective. Both SFT and this form of Supervised KD are fundamentally **off-policy**, as they train on a fixed, static dataset $(X,Y)$.

#### GKD and On-Policy Data

The paper's primary contribution is to address the **train-inference distribution mismatch** that arises from off-policy training. The **On-Policy KD** objective, $L_{OD}(\theta)$, achieves this by altering the data distribution for the expectation: $\mathbb{E}_{x\sim X}[\mathbb{E}_{y\sim p_{s}(\cdot|x)}[\mathcal{D}_{KL}(p_{T}||p_{S}^{\theta})(y|x)]]$.

The inner loss function remains the same (the KL divergence from teacher to student), but the outer expectation is now over sequences $y$ sampled *from the student's own policy* $p_s$. This reframes the problem as **on-policy imitation learning** (akin to DAgger). The student generates a sequence, including its own errors, and the teacher provides "soft" corrective labels for that *same sequence*. This forces the student to learn how to recover from its own mistakes, aligning the training distribution with the inference-time distribution. The authors note this is done without backpropagating through the student's sampling process, ensuring computational stability.

The **Generalized Knowledge Distillation (GKD)** framework is then introduced to unify these approaches. It uses a hyperparameter $\lambda$ to control the mixture of on-policy ($y \sim p_s$) and off-policy ($y \sim (X,Y)$) data.
* When $\lambda=0$, we recover pure Supervised KD.
* When $\lambda=1$, we recover pure On-Policy KD.
* This GKD framework also generalizes the *divergence function* itself, allowing for Forward KL (mode-covering), Reverse KL (mode-seeking), and JSD.

#### RL + KD

Finally, the paper situates GKD within a standard **Reinforcement Learning** context. The on-policy GKD objective is reframed as a **regularizer** for an RL objective: $\mathbb{E}_{y\sim p_{S}^{\theta}}[(1-\alpha)r(y) - \alpha\mathcal{D}(p_{T}||p_{S}^{\theta})]$.

This is a powerful multi-objective formulation. The model is trained to maximize a scalar reward $r(y)$ (the RL goal) while the $\alpha$-weighted GKD term penalizes the policy for diverging from the teacher $p_T$. This KL regularization acts as a constraint, preventing the student from discovering a "pathological" policy that overfits to the reward metric while sacrificing generative quality. This directly addresses the "alignment tax," where models fine-tuned for a specific capability (e.g., factual consistency) suffer a decrease in general performance. The framework is shown to be easily implemented by modifying existing RLHF/RLAIF pipelines, simply by setting the reward to zero and replacing the SFT reference policy with the teacher policy.