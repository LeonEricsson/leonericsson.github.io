---
layout: post
title: ""
categories: []
year: 2024
type: blog
---
Physics of Language Models is a collection of papers that aim to understand how Language models work. In their own words: "*Our goal is to establish universal laws for LLMs that can guide us and provide practical suggestions on how we can ultimately achieve AGI.*". 

Even today, GPT-4 and Llama-3 still provide incorrect answers to some questions that are simple for humans. Is this a problem inherent to GPT-4, or is it due to insufficient training? Is its mathematical capability too weak? Does this only affect models as of July 2024, or will GPT-6 and Llama-5 also face this problem? What about other pre-trained models? 

We propose dividing the concept of "intelligence" into multiple dimensions (such as structures, knowledge, reasoning, etc.). For each dimension, we create synthetic data and build an idealized environment for LLM training to understand the theory and push the capability of LLMs in this dimension to the extreme. By performing a large number of controlled experiments, we aim to discover the universal laws of all LLMs, not just a particular version of GPT-4. 

- Decompose intelligence into building blocks (structures, knowledge, reasoning), and study them individually
- Build synthetic datasets that enable study in controlled, idealized environments. This allows you to tweak parameters such as difficulty, amount, distribution, and evaluate how these things effect model performance. Allows you to make informed decisions on how to train future models
- Highly repeatable experiments. Use smaller models (100M size), where you can run feasible, repetitive experiments, in an attempt to derive universal laws. 
- Probing techniques to see the inner workings of models.

# Part 3

How do Language Models actually store knowledge? Back in novemeber 2023, the authors asked GPT-4 a simple question "Was Joe Biden born in an odd year?" to which it answered Yes. This is of course wrong as Biden was born in 1942. Why does this happen? We can break do down as such

![](/images/physicspart3.png) 

# Part 3.1 Knowledge Storage and Extraction

If we assume that the model has seen Joe Biden's birth year during pretraining, then we can ask ourself why it is not able to extract this information at test time. That is exactly what part 3.1 aims to understand: understand under what conditions language models can extract information that it has seen in its pretraining data. 

![](/images/physicspart31.png) 

Designed a synthetic dataset of biographical data of N individuals, called the BIO dataset, listing information on 6 attributes name, birthdate, education, employment, living locations, etc in varied sentences.

"*Anya Briar Forger was born on October 2, 1996. She spent her early years in Princeton, NJ. She received mentorship and guidance from faculty members at Massachusetts Institute of Technology. She completed her education with a focus on Communications. She had a professional role at Meta Platforms. She was employed in Menlo Park, CA.*"

Paired with these biographies, they design a Q/A dataset, constituting 6 questions per individual. Each question is used as a prompt for the model to generate a response. This dataset is split into a train and test set, such that one can truly evaluate **memorization**: the model has seen the answer during training **extraction**: the model is able to extract the answer to the question from the BIO. 

The model is trained from scratch, and the BIO data is always a part of this pretraining dataset. Sentences are concatenated into 512 token sequences, separated by a standard <EOS> token

Now we've got the setup ready, this is what the authors use to examine the aforementioned question: under what conditions are language models able to extract information that it has been pre-trained on. Remember that the BIO data is **always** a part of the pretraining data, so the answers are all there.

**Mixed-Training**. BIO and Q/A (only the train half) data is randomly sampled and trained from scratch. Under these conditions, the language model is able to accurately answer 86.6% of the test set QAs. Hence, Mixed-Training -> Knowledge extraction

**BIO Pretrain + QA Finetune**. Mixed-training is however not akin to how language models are typically trained. Rather, models are pretrained on randomly structured corpus of web data, and then instruction tuned. To mimic this, the authors pretrain a model on the BIO data and then finetune on the QA data, both full fine tuning and LoRA. Despite a 99% accuracy on both the pretrain and the finetune datasets, the model is unable to generalize out of distribution to the test set QAs. This is a universal law, independant of architecture, size, training parameters etc:

**Rule:** *A model pretrained to word-by-word memorize knowledge may never be fine-tuned to extract knowledge. Perfect BIO token memorization + perfect QA answers for half the people != correct QA answers for the other half.*

This comes as a shock to me. And it did for the authors as well, until they realized that there is only one biography per person. What happens if we augment the BIO dataset such that knowledge is mentioned multiple times using sentence diversity, permutation, translation, writing styles? Well... the test accuracy jumps to **96%**! So, it's absolutely necessary to augment the pretrain data for knowledge extraction. But, why?

To answer this question the authors probe the models, in what they call position-based probing (P-probing). They identify six *special token positions* immediately preceding the first occurrences of the six attributes in each biograph entry:

"*Anya Briar Forger was born **on** October 2, 1996. She spent her early years **in** Princeton, NJ. She received mentorship and guidance from faculty members **at** Massachusetts Institute of Technology. She completed her education with a focus **on** Communications. She had a professional role **at** Meta Platforms. She was employed **in** Menlo Park, CA.*"

Using the hidden state of the last hidden layer at each of these token positions, the authors train a linear classifier to predict the target attributes. Without knowledge augmentation, the P-probing accuracy is close to 0% until the token immediately preceding the target attribute. The authors posit that this happens because the model learns the *wrong logic*: someone born on Oct 2, 1996 in Princeton and studied Communications at MIT works for Meta. Rather than learning the *right logic*: Anya works for Meta. When knowledge augmentation is applied, given that the connection of Anya -> Meta is seen multiple times, in multiple permutations, the model does store this (key, value) correctly. The P-probing supports this, and supports a claim even stronger actually: the accuracy for all six attributes rises to nearly 100% from the first special position, meaning before **all** of the attributes! This indicates that the model not only memorizes the BIO data but also identifies the person's complete six attributes solely upon seeing the person's name, facilitating knowledge extraction during the QA finetuning process.

**Rule:** *Adding multiplicity, permutations, or repeating full names, all help the model to better store knowledge during pretraining, making knowledge extraction easier later.*

In practice, it's not feasible to augment data for all individuals. And you'll most likely find that some individuals appear in several instances in the data, and others only appear once. To explore the effects of this on the model, the authors devise an experiment of N minorities without knowledge augmentation (1 entry), and M celebrities with knowledge augmentation (5 entries). The model is pretrained on the BIO data (N + M), and finetuned on M/2 QAs. According to our previous knowledge, the performance on test celebrities should be high, because we applied knowledge augmentation to the celebrities and half the QAs are in the finetuning data, and the performance of minorities should be bad, because they are not augmented, and are not in the finetune data. But what actually happens is that we get 80% accuracy on the N minority QAs! Wow! It turns out that by mere existence of knowledge augmented individuals in the pretraining data, the model learns **how** to store knowledge in a format that enables knowledge extraction.  

**Rule:** *Introducing celebrity data boosts the minority group’s QA accuracy (e.g., from 4.4% to 86.8% for the bioS data).*

# Part 3.2 Knowledge Manipulation
Knowledge Manipulation assumes that knowledge is fully extractable, as explored in Part 3.1, and now you instead want to study the skills of language models to manipulate the knowledge. For example, can they answer questions such as "Was Joe biden born in an odd year?" or "Was Donald Trump born earlier than Nancy Pelosi?" based on their memorization of celebrities' birthdays? The authors are interested in questions that are *functions* of extracted knowledge from the pretraining data. The *functions* explored in this study are simple forms of logical reasoning, as exemplified earlier: "Is X born in an odd year?", "Was A born in a southern city?" etc. 

To test this, the authors pretrain a model on the BIO data of N individuals, then they finetune or pretrain on knowledge extraction of all N individuals (this is the same as QA finetune from 3.1), and finally they finetune on knowledge **classification** on N/2 individuals. Classification finetuning here means finetuning on examples of manipulation, with and without CoT:

"Q: Was Anya Briar Forger born in an even month? 
A (without CoT): Yes.
A (with CoT): October, so it is Yes."

This setup should look familiar because its the same premise as in 3.1, pretrain on the prerequisite knowledge, finetune on half the examples of the task you want to evaluate, then evaluate on the other half.

So, what happens when we evaluate the model ability to classify the remaining N/2 individuals? When the model is prompted to use CoT, it has a 100% accuracy, but (!) without CoT, the accuracy is no better than random guessing: 50%?? What this means is that the authors discover that knowledge manipulation, even in its **simplest** form, single-step, is impossible without CoT. This applies for other manipulation tasks, ranking, comparison etc. Despite the inclusion of CoT at training time, the model can not manipulate knowledge without first writing it out explicitly (CoT) at inference time. Now, please note that this is different from CoT in reasoning, GPT-4 is very capable of answering what the sum of A and B are without writing down A+B explicitly, but when it comes to manipulating extracted *knowledge*, it has to write down the knowledge before manipulating it.

Even though these laws are established on experimental models, these rules did apply even to the best and brigtest LLMs (at least as of July 2024). Asking GPT-4 to classify birth months of celebrities it has 50.7% accuracy, asking it to rank birth dates it has 52.3% accuracy.

# Part 3.3 Knowledge Capacity Scaling Laws

Scaling Laws are a pivotal area of research in large language models as they enable predictions about the performance through experiments with smaller ones. In the training regime there are established models for these scaling laws, discussing the optimal training flops versus model size. However, there is a lack of research in trying to understand what the *ultimate* performance that a model is capable of achieving, assuming sufficient training. This study aims to understand scaling laws of knowledge capacity, that is, estimating the number of knowledge *bits* a model stores, with a focus on factual knowledge representation as tuples, such as (USA, capital, Washington D.C).

So what do we mean with a knowledge bit?

Let's imagine creating a synthetic English dataset describing knowledge tuples, and creating our first entry:

- (Anya, birthday, 10/2/1996)

Well if these birthdates are uniformly drawn from any day, of any month, in the past 200 years, then this represents $\log_2(12 \times 28 \times 200) = 60.21$ bits of knowledge. So going back to the BIO data consisting of N biographies, we can actually estimate the amount of information that is present, and that is regardless of how the biographies are written, and how many times they are repeated. Now, in the paper they introduce a new synthetic dataset that has even more tunable hyperparameters than just N biographies but I'm not going to detail this because the logic is the same: *for any time of synthetic knowledge, the authors propose a formula to compute the knowledge bits stored in this dataset*.

Now suppose we pretrain a LLM on your synthetic dataset, we can actually compute the amount of learned knowledge (in *bits*) of our models. However, doing so it not trivial. If the model achieves 0 loss on the dataset, then obviously the model has fully captured the knowledge, but what if the model is 50% accurate? Well in that case you need to be more careful in how many bits you say that the model stores.

The authors posit that for a wide range of model sizes/depths / widths (i.e. only size matters):

**LLM can "consistently" achieve 2bit/param in storing knowledge if sufficiently pretrained** 

but what does *sufficiently trained* mean? It means that the language model has to be exposed to the knowledge (think knowledge tuple from before) 1000 times. This doesn't necessarily mean that you have to pass over every piece of knowledge 1000 times, but rather that you have to see the tuple, in some constellation (different writing styles are ok), 1000 times. 

but what if there are less exposures? Well then GPT-2 consistently achieves 1bit/param, but even more interesting is that at 100 exposures then the architectures start to matter! In the 100-exposure setting, some architectures are worse in knowledge capacity: e.g., LlaMA / Mistral architectures can be 1.3x worse than GPT. Don't forget that we're talking about the **knowledge capacity** of these architectures, and specifically knowledge capacity for rare knowledge. Controlled experiments by the authors find why this is the case, there are in total 7 architectural differences between GPT-2 and LLaMA, and by turning each one of these on and off they find that its the MLP layer that causes this difference. Replacing the GatedMLP layer of LLaMA / Mistral with vanilla MLP, then the knowledge capacity improves to that of GPT-2 at 1bit/param. 