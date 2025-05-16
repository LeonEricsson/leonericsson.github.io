---
layout: post
title: "how transformers think"
categories: []
year: 2024
type: blog
---
<<<<<<< HEAD
Mechanistic Interpretability is a tool to understand the internals of neural networks, specifically in this case transformer language models. Note that I'm saying **a** tool, not **the** tool. As any tool it has limitations, and in MI that usually involves looking at models through a microscope and trying to analyze certain, but limited parts of the behaviour in an attempt to create a broader picture of what is happening. It is important to understand that much work in MI is limited in this sense where behaviour can look present through studies but that doesn't account for every example tested. MI will often highlight the success cases where the authors actually managed to learn something interesting. The findings are often of the type of "existence proof" - concrete evidence that specific mechanisms operate in certain contexts. While you may suspect similar behavior in general this is where you cross the line of evidence into speculation. You typically have to examine a small subset of the entire model, either in a simplictic or reduced form to be able to interpret anything, but this always risks introducing bias. 

## Framework for understanding the transformer
When I talk about the transformer in this blog I am referring to an autoregressive, decoder-only transformer language model. There are **a lot** of variants and modifications to this architecture, some more relevant than others, but the core blocks have remained the same for significant duration of time. It starts with a token embedding, followed by a series of layers or residual blocks as I prefer to call them, and finally a token unembedding layer. A residual block consists of an attention layer, followed by an MLP layer. The token (un)embedding layers transform a token, a single integer, into a *d*-dimensional float representation and vice versa.

To start, the embedding matrix transformers a sequence of $n$ tokens

$\mathbf{t} \;=\; (t_1,\,t_2,\,\dots,\,t_n) \;\in\; V^{\,n}$

into an embedding

$h \in \mathbb{R}^{\,n \times d}$ 

but for simplicity I want to focus on the final embedding in the sequence as it translates into the next token prediction. So we will focus on $h_n \in \mathbb{R}^{d}$ but I will omit the subnotation $n$. Let's have a look at this vector traveling through our transformer.

![alt text](image.png)

I really want you to focus on the residual stream here. Ignore exactly what's going on in the attention or MLP layer for now. Just focus on the flow of information. Notice how every layer in this architecture reads and writes to the residual stream. The residual stream doesn't do any processing itself, it acts as a communication channel for the layers. Every layer performs an arbitrary linear transformation to "read in" information from the residual stream and the start and performs another linear transformation before adding to "write" its output back into stream.

There are several very interesting effects of this linear structure.
The residual stream is a high-dimensional vector space. This means that layers can send different information to different layers by storing it in subspaces, without even interacting with intermediate layers. Once information has been added to a subspace it will persist unless actively deleted by another, this means that dimensions of the residual stream become something like a memory. 

Remember that we are only looking at the embedding at the single, final position in the sequence. And a natural question would be how information travels between positions in the sequence. This is where we get into the operations of the MLP and the Attention layers. Attention layers, or specifically attention heads, fundamentally act to move information between positions in the sequence. They read information from the residual stream of one token and write to the residual stream of another. Importantly, which token to move information from is completely separable from what information is "read" to be moved and how it is "written" to the destination. We divide the attention into two linear operations $A$ and $W_o W_v$, which operate on different dimensions and act independently.

- $A$ governs which token's information is moved from and to.
- $W_oW_v$ governs which information is read from the source token and how it is written to the destination token. 

The MLP layer operates independently on tokens.  

The token embeddings, as well as the unembeddings interact with a small fraction of the total number of dimensions. This means that most of our dimensions, and our vector is free to store information in. 

## Do Llamas work in English?
An mechanistic investigation into the latent language of multilingual transformers. Language models are primarily trained on English text, with a heavily biased distribution. Despite this they achieve strong performance on a broad range of downstream tasks, even in non-English languages. How do they achieve this generalization? This paper analyzes this through two distinct views. First, the premise, consider a translation task from french to chinese with the following few-shot prompt (where "中文" means "Chinese"):

```
Français: "neige" - 中文: "雪"
Français: "montagne" - 中文: "山"
Français: "fleur" - 中文: "
```

with the guarantee that the expected chinese translated word is a single-token word. This setup allows us to solely analyze the latent at position $n+1$, as it carries all the information necessary to produce the chinese translation.

The first method of analysis is a logit lens. A logit lens approach means that we take the models unembedding matrix $U$, which is normally applied after the final layer to get the logits which are turned into a next-token distribution, and instead apply after an intermediary layer $l$. This gives us an inspection tool to look at what tokens the model thinks is most probably as the latents travel through the model! This is very cool. Let us track the probabilities of the English translation word and the Chinese translation word throughout the layers, are we using english as a pivot language when performing this translation?

We find for the first half of the models layers, neither the chinese or the english analog garner any noticeable probability mass, then around the middle layer, english begins a sharp rise followed by a decline, while chinese slowly grows and, after a crossover with english, spikes on the last five layers. Wow, that is really cool, it does seem like english is used as a sort of pivot language. Looking at the entire next-token distribution throughout the layers, through entropy, we see that the first half layers is a uniform distribution with an entropy of ~14 bits (vocab is 32,000 so this makes sense) and then, at the same time as the english analog spikes in probability, the entropy sharply drops and remains low until the end of the model. 

Now, this is analysis provides a probabilistic perspective, studying the next-token distributions via the logit lens. The other promised perspective is that of a geometric one where we analyze the latens directly as points in Euclidean space. 

When you think of what a transformer is doing, you should be thinking of that of a vector moving through *d*-dimensional space. If we break it down. Following the embedding matrix, the starting position, before anything has happened in the model, is a latent / embedding 

$h $
=======
-- placeholder --

## A transformer framework
Within the context of this discourse, a *transformer* denotes an autoregressive, decoder-only transformer language model. Numerous variants and modifications to this foundational architecture exist, some exhibiting greater relevance than others; yet, the fundamental architectural constructs have demonstrated remarkable stability over time: a token embedding layer $\rightarrow$ a sequence of layers known henceforth as residual blocks $\rightarrow$ a token unembedding layer. Each residual block canonically comprises an attention mechanism succeeded by a multi-layer perceptron. The token embedding and unembedding layers are responsible for mapping a token—represented as a discrete integer identifier—to and from a $d$-dimensional floating-point vector representation.

The embedding matrix operates on an input sequence of $n$ tokens:

$\mathbf{t} = (t_1,t_2,\dots,t_n) \in V^{n}$

transforming it into an embedding matrix:

$H \in \mathbb{R}^{n \times d}$

where $H = [h_1, h_2, \dots, h_n]^T$ and each $h_i \in \mathbb{R}^{d}$ is the embedding for token $t_i$. However, for conceptual clarity, our analysis will concentrate on the embedding vector at the final sequence position, $h_n \in \mathbb{R}^{d}$, as this represents the next token prediction. Henceforth, the subscript $n$ on $h$ will be omitted for brevity, assuming we are referring to this final-position embedding, unless explicitly required for disambiguation. Consider the trajectory of this vector, $h \in \mathbb{R}^{d}$, as it propagates through the transformer's layers.

[Image]

The most important concept to grasp here is the **residual stream**. For the present, defer a detailed examination of the internal operations within the attention and MLP layers. Instead, concentrate on the overarching information flow. Observe that each layer within this architecture interfaces with the residual stream by both **reading** from and **writing** to it. The residual stream, intrinsically, performs no computation; rather, it functions as a high-bandwidth communication bus facilitating inter-layer information exchange and state propagation. I find this to be one of the most insightful ways of thinking about the transformer. Each layer applies a learned linear transformation to project information from the incoming residual stream into its operational space (a "read" operation). Subsequent to its internal computations (e.g., attention or MLP), it employs another learned linear transformation to project its output, which is then additively merged back into the residual stream, forming the input for the subsequent layer (a "write" operation).

This linear, additive structure of the residual stream has some really cool and important consequences. Being a high-dimensional vector space (think of it as having lots of "lanes"), the residual stream allows distinct layers to communicate information selectively by encoding it within orthogonal or non-interfering subspaces. This means information can take an express route, effectively creating direct communication pathways across network depth. Once information is added to one of these subspaces, it tends to stick around through the following layers unless another layer specifically changes or removes it. Consequently, dimensions within the residual stream can be conceptualized as persistent memory slots, accumulating and refining information as it traverses the network.

A natural question arises: how is information propagated from prior token positions to this final position, which is critical for contextual understanding and next-token prediction? This necessitates an examination of the MLP and, more critically, the attention layers. While MLPs operate on each position independently, refining the representation at that position, attention layers, or more granularly, individual attention heads, serve the primary function of transposing and integrating information between different token positions.

Having established the residual stream as the primary conduit for information propagation, the natural question is: what actually happens *inside* these residual blocks? Specifically, what kind of processing are they doing with the information they've just read from the stream? Recall the two modules inside the residual block: **Attention** and **MLP**.
>>>>>>> 81d53179318b4aaf8b0252160b8ceb8f9ffbce04


Attention facilitates inter-token communication, enabling the model to dynamically integrate information across different sequence positions. The mechanism operates by selectively aggregating information from the residual streams corresponding to a set of 'source' token positions and writing some processed version of that information to the residual stream of our "target" token. What's really neat is that how the model decides which tokens to get information from is pretty separate from what information it pulls out and how it changes that information before adding it to the target token's stream. Consequently, an attention head's operation can be effectively decomposed into two substantially independent computational sub-circuits:

The **Query-Key (QK) circuit**: This computes the attention pattern, typically a normalized score matrix $A \in \mathbb{R}^{n \times n}$ (for self-attention focusing on the final token's representation, we are interested in the $n$-th row, $A_{n,:}$, which dictates the contribution of each sequence position $j$ to the current position $n$). Essentially, it decides which other tokens' information is most important to pay attention to for the current token.

The **Output-Value (OV) circuit**: This circuit first projects the input token representations $h_j$ into 'value' vectors $v_j$ using a weight matrix $W_V$. It then combines these value vectors $v_j$ according to the attention pattern computed by the QK circuit. Finally, this aggregated information is projected using an output weight matrix $W_O$. Thus, the composite operation, effectively $W_O \left( \sum_{j} A_{n,j} (h_j W_V) \right)$, governs what information is extracted from the attended tokens and how it is transformed and integrated back into the residual stream at position $n$.

*Note: The initial token embedding and final unembedding layers typically engage with only a subset of the $d$ dimensions of the residual stream. This leaves a significant portion of the dimensional capacity of the residual stream available for internal computations, feature learning, and information storage by the intermediate transformer blocks.*

## How do transformers think. Yes, they are thinking.
Now that we've established a common framework for how information flows **through** the transformer. I know want to present my mental model for thinking about how transformers think. And when I say think I mean think, a transformer is thinking. 

My mental model posits that a transformer "thinks" by manipulating information within a high-dimensional, largely language-agnostic **conceptual manifold**. This processing occurs primarily *off-stage* from direct token representations. However, this conceptual stage is built upon a foundation heavily influenced by English, leading to an observable "English-centricity" when we try to peek at the intermediate steps through the lens of token probabilities. The **residual stream**, a concept central to your framework, is the lifeblood of this process—a dynamic blackboard where concepts are written, refined, and composed.

The journey of a thought through the transformer begins with **de-tokenization, an escape from the token cage** orchestrated by the early layers. The initial token embedding is a necessary evil, a conversion of discrete human symbols into the model's native language of dense, $d$-dimensional vectors. Each such vector, the hidden state $\mathbf{h} \in \mathbb{R}^d$, represents the model's evolving understanding at a particular point in the sequence. Crucially, as highlighted by research like Anthropic's work on "de-tokenization layers" and the "token energy" analysis from studies like "Do Llamas Think in English," these early layers actively work to move the representation *away* from the specific input tokens. The model's ultimate task is to predict the next token, a process mediated by an unembedding matrix $U \in \mathbb{R}^{v \times d}$ (where $v$ is vocabulary size), whose rows are vectors representing each token. These token vectors span a subspace within $\mathbb{R}^d$, which we can call the **token subspace $T$**. Only the component of the hidden state $\mathbf{h}$ that lies within this token subspace, let's call it $\mathbf{h}_T$, directly influences the output probabilities (logits). The remaining component, $\mathbf{h}_{T^{\perp}}$, is orthogonal to $T$ and thus invisible to the final output layer. In these initial layers, $\mathbf{h}$ rapidly becomes largely orthogonal to $T$; that is, its $\mathbf{h}_{T^{\perp}}$ component dominates, and its "token energy"—the extent to which $\mathbf{h}$ aligns with $T$—is low. My contention is that the transformer isn't just finding a vector for "cat"; it's unpacking "cat-ness" into a richer, more flexible representation within this conceptual manifold, primarily residing in $\mathbf{h}_{T^{\perp}}$. It's shedding the arbitrary boundaries of tokenization (e.g., "ice" and "cream" vs. "icecream") to access a more fundamental semantic space. This initial phase is about creating a sufficiently abstract representation ripe for genuine computation, moving away from the direct influence of the token subspace $T$.

Following this escape, the **conceptual alchemy—the silent thought in the orthogonal deep**—takes place in the middle layers. This is where the *real* thinking happens. The middle layers operate predominantly within the vast space of $\mathbf{h}_{T^{\perp}}$, the conceptual manifold invisible to the immediate logit lens. Here, the residual stream acts as a compositional workspace. Information, now in the form of abstract features, flows through this stream. Each processing block, comprising attention and MLP sub-layers, reads from this stream, performs its specialized computation, and writes its refined output back. The MLPs act as feature refiners at each position, deepening the understanding of the local conceptual context. They are the introspective engines, taking a concept represented in $\mathbf{h}_{T^{\perp}}$ and making it more precise or nuanced. Attention mechanisms, on the other hand, are the relational engines. The Query-Key (QK) circuit isn't just finding relevant prior tokens; it's identifying which *abstract conceptual features*—elements of $\mathbf{h}_{T^{\perp}}$ from other positions—are most pertinent to the current conceptual state. The Output-Value (OV) circuit then doesn't just copy information; it *transforms and integrates* these selected conceptual features, effectively performing complex cognitive operations like analogy, comparison, or causal inference within this abstract $\mathbf{h}_{T^{\perp}}$ space.

This framework accommodates the intriguing findings on multilingualism and the often-discussed English privilege. Anthropic's research on "multilingual circuits" is key: concepts like "antonym" or the essence of "smallness" exist as language-independent features in this conceptual manifold. The model can reason about "the opposite of X" without committing to a specific language for "X" or its antonym during this core processing within $\mathbf{h}_{T^{\perp}}$. However, the "English tilt" observed across various studies is undeniable. My argument is that this is not because the transformer "thinks *in* English words," but because the conceptual manifold itself has been shaped by the sheer volume of English in its pre-training. English token vectors (the rows of $U$ corresponding to English words) form a particularly dense and well-structured lattice within the token subspace $T$. Consequently, when a concept evolving in $\mathbf{h}_{T^{\perp}}$ begins to develop components that project into $T$ (i.e., its $\mathbf{h}_T$ component starts to gain energy and significance), it's statistically and energetically "easier" for this projection to initially align with a nearby, dense English anchor within $T$. This is an attractor effect. It's a path of least resistance for the projection $\mathbf{h}_T$, especially for lexical content words. This explains why, when viewed through the logit lens (which only sees $\mathbf{h}_T$), English equivalents often appear first, as observed by PPL @ UW et al. It's a shadow of the underlying computation in $\mathbf{h}_{T^{\perp}}$, not the computation itself. The steering vector results (PPL @ UW 4.2) further support this: English-derived steering vectors are more effective because they are better aligned with this dense "English lattice" in $T$, allowing them to guide $\mathbf{h}_T$ along "ridges" of high logit sensitivity for the desired concept, regardless of the final target language.

As the computation nears its end, the phase of **re-tokenization and articulation—projecting thought into language**—commences in the late layers. The abstract conceptual representation refined within $\mathbf{h}_{T^{\perp}}$ must now be translated back into the world of discrete tokens. This is where "token energy" spikes, signifying that $\mathbf{h}$ decisively projects onto the token subspace $T$, meaning the magnitude of $\mathbf{h}_T$ becomes dominant. The "re-tokenization layers" identified by Anthropic come into play. Anthropic's work also suggests the presence of language-specific features that guide this final output. If the target language is not English, the hidden state's projection $\mathbf{h}_T$, which might have transiently flirted with an English anchor due to the aforementioned "tilt," now actively moves towards the correct target-language anchor within $T$. This might involve a more significant "angular cost" if the target language anchors are sparser in $T$, but it's a necessary step for accurate translation. This final shift explains how the model can entertain a concept that momentarily "looks like" English via the logit lens, but then output it correctly in Mandarin. The core concept was largely language-agnostic (though developed in an English-tilted conceptual space); the final layers handle the culturally and linguistically appropriate surface form. For grammatical function words, which PPL @ UW et al. note rarely appear as English equivalents first, the story is simpler: these often form their own dense "islands" within $T$ for each language, so $\mathbf{h}_T$ can navigate directly to, say, a Dutch determiner without an English detour.

<<<<<<< HEAD
![](/images/featureoverlap.png)
![](/public/images/featureoverlap.png)

Looking at the beginning and end of models they tend to be language specific, consistent with the de/re-tokenization layer hypothesis.

**De-tokenization layers.** The early layers of a language model are involved in mapping the artificial structure of tokens to a more natural semantically meaningful representation. Many early neurons respond to multi-token words or compound words. 

**Re-tokenization layers**. In reverse, the final layers of language models mediate conversion of words or contextualized tokens back into literal tokens of the vocabulary. There are neurons which convert or dictate a representation of the word into constituent tokens one by one for output. 

This is why depth is important. This coincides with the token energy discussion seen before where the overall latent vector is mostly orthogonal to the token space throughout the early -> middle layers. While the features represented in the middle layers are multilingual they are mostly aligned with english as a result of biased pretraining. English is privileged in a way where the default output is English but the representation space is large, even though English is privileged doesn't mean models think in English.

## Do Llamas think in English
To complement the probabilistic view, let us explore a geometric perspective, analyzing the latens directly as points in Euclidean space,   i.e before mapping them to token probabilities.

Simplistically the task solved by an autoregressive model is that of mapping the input embeddings of the current token to the output embeddings of the next token through incremental modification of the latent *d*-dimensional vector traveling through the transformer. Geometrically, this latent describes a path through *d*-dimensional Euclidean space. We set to characterize this path. 

Every forward pass of a language model produces, at each position, a latent vector $\mathbf h\in\mathbb R^d$ whose dimensionality $d$ is fixed by model architecture. At the output end, the model applies a single unembedding matrix $U\in\mathbb R^{v\times d}$, where each row corresponds to a vocabulary token. The logits emerge purely as inner products: $\mathbf z = U\mathbf h$. No bias terms or nonlinearities intervene between the final hidden state and the logits.

The effect of this structure is that only directions spanned by the rows of $U$ - that is, the token vectors - can influence the logits. This set of directions defines a **token sub-space**


$$
T \;:=\; \operatorname{span}\{u_t \;|\; t\in V\}
     \;=\;\Bigl\{\,
         \sum_{t\in V} \alpha_t\,u_t
         \;\Bigm|\; \alpha_t\in\mathbb R
       \Bigr\}\subseteq \mathbb R^{d}.
$$
Algebraically, the token vectors are full-rank, so $T$ spans all of $\mathbb R^d$, but empirical analysis show that token vector concentrate on a low-dimensional ellipsoid. 

This is where things get interesting. We can decompose our latent uniquely as 

$$
\mathbf h \;=\; \underbrace{\mathbf h_{T}}_{\text{lies in }T}
            \;+\; \underbrace{\mathbf h_{T^{\!\perp}}}_{\text{orthogonal to }T}
$$

and means that a computation of our logits $\mathbf z$ can be written as

$$
\mathbf z = U\mathbf h \;=\; U(\mathbf h_{T}) \;+\; U(\mathbf h_{T^{\!\perp}}) \;=\; U\mathbf h_{T},
$$
showing that logits depend only on the component that sists inside the token sub-space; anything in the orthogonal directions is invisible to the language-model head. 

This observation allows a clean decomposition. The latent $\mathbf h$ separates uniquely into two orthogonal components: one inside $T$, the other in its complement $T^{\perp}$. The projection onto $T$ can be written as $P_T = U^+U$, and logits depend solely on the projected part: $U\mathbf h = U\mathbf h_T$.

Remember back to our previous discussion on the anatomy of a language model, with the residual stream being a communication channel of low dimensional subspaces.

The logit lens is inherently blind to $\mathbf h_T$ meanwhile the orthogonal component may be important to the computations carried out by later layers and for predicting the next token in those layers. Crucially the logit lens only sees part of the picture, a projection of the true *d*-dimensional latent onto a token plane. To qualify this, the authors establish *token energy*, a measurement which captures how much of $\mathbf h$'s "energy" translates into logit scores. By analyzing this metric throughout the layers they find that token energy remains relatively low throughout the first 40 layers, then as entropy collapses and the english language probabilities dominate, the token energy increases but only slightly, before spiking on the last 10 layers when the model probalities switch from english to chinese.

What's going on here? Initial probing made it seem like english was used as a pivot language in the models middle layers before finally translating to the desired language. Further analysis shows us something even more interesting. In fact our latent contains very little information of the output tokens. However, probably due to the natural language bias of pretraining, we're slightly biased towards english output tokens. What is actually happening in these middle layers is beyond scope of the paper but we know out latents are carrying a lot of information that is orthogonal to token subspace. Imagine 2 dimensions where the x plane represents output tokens, we can imagine that the our model is freely manipulating the y dimension of this vector, manipulating information as the latent flows through the model. In the early layers there is zero movement along the x-plane and the responding probabilities are 0, as we pass layer 40, the model is still manipulating most of its information in the y dimension, but projected onto x there is now a slight bias towards the english output tokens. Then as we get closer to the final layers, our x dimension shifts and most of our token energy now comes from the x dimension and now near the chinese language tokens.

### Do Multilingual LLMs Think In English?

#### 4.1
Prompt the model to generate full sentences in Dutch, French, German and Mandarin and decode every intermediate latent. A broad analysis of the latents, through logit lens, shows that lexical tokens (noun, verbs, pronouns) overwhelmingly appear in their English equivalents first; grammaticl function words rarely do. Semantic decisions are made in a representation region closest to English; whether the input/output language is Dutch or Mandarin is secondary. Degree of routing correlates with pre‑training multilingual diversity and model size.

#### 4.2
Build topic and language steering vectors from parallel corpora. Compare sucess rate when steering non-English prompts with English-derived vectors vs target-language vectors. English steering vectors consistently outperform same‑language vectors at inducing the desired topic and keeping fluency. Steering vectors across languages have high cosine similarity but retain a language‑specific offset. If the concept space were truly universal, steering effectiveness would be language‑agnostic. Superior English steering indicates the latent space is English‑centric, not language‑neutral.

#### 4.3
Use a city facts dataset containing translations of city facts such as "The capital of Canada is .." / "De hoofdstad van Canada is ...". Pinpoint the layers where these facts are stored through causal tracing; then linearly interpolate between the Dutch and English hidden states. Facts localise in the same layers regardless of language. Interpolating hidden states keeps the answer (“Ottawa”) correct while the output language drifts toward English, showing an English decoding bias. Semantic content appears shared across languages (supporting a common fact manifold), but the decoding head pulls outputs toward English, reinforcing the claim that key decisions live in an English‑tilted region.

#### Relation to "Do Llamas think in English" paper
Despite arguing against Wendler et al I believe that this paper enriches our understanding of the latent representation established before rather than contradicting them. Taken together, the two papers support a unified picture: latent computation happens in a largely language‑agnostic “concept manifold” that is tilted toward English anchors, so the small part of the vector visible to the logit lens often points to English first—especially for lexical content—but the bulk of the vector is still outside the token sub‑space and remains concept‑level.

A response to each section or how to tie it into the overarching framework:

**4.1** English content tokens dominate the training data, their output‑embedding vectors saturate the token plane with a dense lattice of “anchors.” During generation the model’s latent state $\mathbf h$ starts mostly outside that plane; as soon as its small token‑aligned slice $\mathbf h_T$ begins to grow, gradient‑shaped residuals nudge it toward the nearest high‑density region—usually an English content anchor—because that tiny rotation yields the greatest logit boost per unit movement. Only after the concept is stabilised and token energy has risen do later layers pay the extra angular cost of sliding $\mathbf h_T$ across the plane to a sparser Dutch (or Mandarin) anchor, producing the target‑language word. Function words behave differently: each language’s determiners, prepositions, affixes, etc. form their own dense hubs, so when the network is writing Dutch it can remain in a Dutch island of the token plane and never detour through English. Crucially, this whole English‑first phenomenon concerns the *direction* of the small visible slice; 70–80 % of the latent norm still lives in the orthogonal concept manifold and is invisible to the logit lens, reminding us to interpret these directional peaks only in tandem with token‑energy magnitude.

**4.2** What actually flips the logits is the dot‑product $u_t^{\!\top}(\mathbf h+\Delta\mathbf h)$, i.e. the overlap between $\Delta\mathbf h_T$ and the dense web of token anchors.  Because English anchors tile the plane much more densely than low‑resource anchors, a vector derived from English sentences inevitably contains a **larger** and **better‑aligned** $\Delta\mathbf h_T$:

* High density ⇒ many anchors with large cosine to $\Delta\mathbf h_T$ ⇒ big logit increase for the desired topic words *in any language* after the late‑layer rotation to the target surface form.
* Low density (Dutch, Mandarin) ⇒ fewer good matches ⇒ smaller logit shift for the same ‖Δh‖.

Hence an English steering vector can move the projection $\mathbf h_T$ along a “ridge” of high logit sensitivity, producing a strong topical pull while requiring only a modest perturbation—so fluency is preserved.  Steering vectors built in Dutch must push $\mathbf h_T$ into a sparser region, needing a larger magnitude that disrupts grammatical planning and hurts fluency.

**4.3** Again, in accordance with 4.1, this holds under our framework. 


In conclusion the following unified framework holds:

1. Latent $\mathbf{h}$ starts almost orthogonal to $T$.

2. During processing it acquires concept components $\rightarrow$ projection slips into $T$ near an English anchor (Phase 2, English logits spike).

3. Final layers translate concept to surface form by sliding the projection within $T$ to a target-language anchor (Phase 3, token energy rises; correct token wins).
=======
Throughout this entire process, the **residual stream's enduring role** is paramount. Its additive nature ensures that conceptual features, once written into this high-dimensional vector space, persist unless actively overwritten or modified by subsequent layers. This allows for the accumulation and refinement of meaning across the network's depth. The high dimensionality of the residual stream is not merely a passive conduit; it allows different "types" of information—core semantics, stylistic nuances, the intended linguistic target—to potentially occupy non-interfering subspaces, all contributing to the final state $\mathbf{h}$. The initial token embedding and final unembedding might only engage with a subset of these dimensions, leaving ample "cognitive workspace" within the intermediate layers for these complex, abstract manipulations predominantly within $\mathbf{h}_{T^{\perp}}$.
>>>>>>> 81d53179318b4aaf8b0252160b8ceb8f9ffbce04

In essence, the transformer, as I conceptualize it, is an engine for abstract conceptual manipulation. It builds complex thoughts in a space largely hidden from direct token-level scrutiny ($\mathbf{h}_{T^{\perp}}$). Its "native tongue" is this abstract feature space. The observed English-centricity is a consequence of its upbringing (training data) and the resulting structure of its token output interface ($T$), making English a sort of "first port of call" for projections of thought, rather than the inherent language of its internal reasoning. The true "thought" is more universal, more abstract, and only dons specific linguistic clothes at the very end of its intricate journey through the network's layers. Depth is absolutely essential for this entire process: from shedding token identities, to engaging in complex conceptual reasoning, and finally, to articulating that reasoning in a coherent sequence of tokens.
