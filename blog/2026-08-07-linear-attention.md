---
layout: post
title: linear attention and its gated descendants
categories:
  - Architecture
year: 2026
type: blog
---
a brief history of linear attention, starting from softmax attention we'll walk through the progression into modern variants of linear attention like GDN and KDA.

## softmax attention
Consider ordinary casual softmax attention. For a sequence of length $L$ and head dimension $d$, ignoring the usual scaling factor, we compute

$$
O = \operatorname{softmax}(QK^\top + M)V
$$

where $Q,K,V \in \mathbb{R}^{L \times d}$, and $M$ is the causal mask. 

The expensive part is the token-token interaction. Computing $QK^\top$ produces an $L \times L$ matrix, and multiplying the attention weights by $V$ has the same sequence-quadratic cost:

$$
\underbrace{QK^\top}_{O(L^2d)}
\qquad
\underbrace{\operatorname{softmax}(QK^\top+M)V}_{O(L^2d)}
$$

So attention takes $O(L^2d)$ arithmetic (FA avoids materializing quadratic attention matrix).

The problem materalizes slightly differently under autoregressive inference. With a KV cache, keys and values aren't recomputed but every new query still has to compare against the keys already in the cache and use the resulting weights to read their values. At step $t$, that's $O(td)$ work, and the cache itself grows as $O(td)$.

There's a simple algebraic observation that makes linear attention appealing. Suppose for a moment that we didn't have softmax or causal masking. We could reassociate

$$
(QK^\top)V \quad\longrightarrow\quad Q(K^\top V)
$$

The left-hand side constructs an $L \times L$ matrix. On the right, we first construct a $d \times d$ matrix. If $L$ is much larger than $d$ (typically in the range of 4-8k) this is favorable. 

Of course, we can't just move the parentheses in softmax attention. Both softmax and the causal mask act on pairwise token-token scores. The trick of linear attention is to change the attention computation so that this reassociation becomes possible, while handling causality through an accumulated state rather than a full attention matrix. This isn't algebraically equivalent, we're changing the operation in favor of speed/memory.

## linear attention as a recurrent memory

The reassociation trick is promising, but we need to remember to deal with causality. Dropping softmax is one thing. We still can't let a token attend to positions that haven't occurred yet.

So rather than looking at the entire sequence, let's write out the output for a single token at position $t$. I'll use column vectors $q_t,k_t,v_t\in\mathbb{R}^d$ for one attention head. If we remove softmax but keep causal attention, the simplest version becomes

$$
o_t = \sum_{i=1}^{t}(q_t^\top k_i)v_i
$$

This is familiar. We compare the current query $q_t$ with each key $k_i$ seen so far, use the dot product to weight its value $v_i$, and sum the results. The difference from softmax attention is that these weights are no longer exponentiated and normalized. For now, I'll stick to this unnormalized dot-product form; practical linear-attention formulations can introduce feature maps and normalization, but those aren't needed to understand the trick.

As written, though, we haven't gained anything. Every new query still seems to require going back over all previous keys and values. So where does the reassociation help?

Notice that $q_t$ is the same in every term of the sum. We can regroup the expression as

$$
\begin{aligned}
o_t
&= \sum_{i=1}^{t} v_i(k_i^\top q_t) \\
&= \left(\sum_{i=1}^{t}v_ik_i^\top\right)q_t
\end{aligned}
$$

This is the causal version of the $Q(K^\top V)$ idea from before, now written for column vectors. Instead of building a single $K^\top V$ from *all* the tokens, including future ones, we build the equivalent key-value summary only from the prefix up to position $t$. Focus now on the matrix inside the parantheses. All the past keys and values have been combined into one $d\times d$ matrix. Let's give it a name:

$$
S_t = \sum_{i=1}^{t}v_ik_i^\top
$$

Now the output is simply $o_t=S_tq_t$. If we already had the matrix $S_{t-1}$ from the previous token, computing $S_t$ is especially easy: we just add the contribution from the new token.

$$
\boxed{S_t=S_{t-1}+v_tk_t^\top,\qquad S_0=0}
$$

$$
\boxed{o_t=S_tq_t}
$$

**That's where the recurrent update comes from.** It's not an additional operation we're inventing; it's just a way of maintaining the causal prefix sum without reconstructing it from scratch for every query. Rewriting the operation like this is known as the recurrent form of linear attention. 

For example, after two tokens we have $S_2=v_1k_1^\top+v_2k_2^\top$. When the third token arrives, we add $v_3k_3^\top$ and read the result with $q_3$. There's no need to retrieve the first two keys and values individually again.

I find it useful to think of $S_t$ as a **key-value associative memory**. Each token writes an outer product $v_tk_t^\top$ into it; a query reads from the resulting matrix. The state has $d^2$ entries, and both the rank-1 write and the matrix-vector read cost $O(d^2)$, regardless of how long the prefix is.

This is also why people describe linear attention as a *linear RNN*. It's recurrent because the next state is computed from the previous state, and *linear* refers to that dependence on the state. It doesn't mean the whole model is linear in its input tokens.

Of course, compressing an arbitrarily long history into a fixed-size matrix has consequences for what the model can remember. We'll return to that when we get to the gated variants. First, let's see what this recurrence buys us computationally.


### inference under the recurrent form

Quite a lot, at least asymptotically. With softmax attention, each new token reads an increasingly long KV cache. With recurrent linear attention, each new token performs just two operations:

$$
\underbrace{v_tk_t^\top}_{O(d^2)}
\qquad\text{and}\qquad
\underbrace{S_tq_t}_{O(d^2)}
$$

The work per generated token is now $O(d^2)$, independent of how many tokens came before it. Across $L$ generated tokens, that's $O(Ld^2)$ rather than $O(L^2d)$.

The same shift happens for the memory:

$$
\begin{aligned}
\text{softmax KV cache:} &\quad O(Ld)\\
\text{linear-attention state:} &\quad O(d^2)
\end{aligned}
$$

These are per-head comparisons, leaving aside common implementation details. The important distinction is that the recurrent state doesn't grow with context length. Of course, $O(d^2)$ isn't automatically cheaper than $O(Ld)$ for every $L$, and real inference speed depends on the kernels and the rest of the model. But importantly we've managed to remove the context-length-dependent scan altogether.

Okay, victory? Decoding looks great, memory is constant in sequence length and even the total arithmetic is linear. Unfortunately while decoding is a straightforward win, training/prefill is not.

## training

Suppose we implement the recurrence exactly as written, one token at a time:

$$
S_1\rightarrow S_2\rightarrow\cdots\rightarrow S_L
$$

Each step consists of a rank-1 outer product and a matrix-vector multiplication. The arithmetic cost is still $O(Ld^2)$, which looks attractive next to softmax attention's $O(L^2d)$. But GPUs despise doing a long sequence of tiny operations. There's two clear issues here:

- Dependency chain length is $L$, so parallelism is poor in the naive recurrent implementation.
- The operations are many small rank-1 outer products ($v_tk_t^\top$) and matrix-vector products ($S_tq_t$), which are inefficient compared to dense matmuls.

While algorithmically fewer in FLOPs we're making poor use of existing hardware. 

### chunkwise parallel linear attention
The problem isn't really the amount of computation we're doing, but how we're organizing it. We have a sequence of small, dependent operations, whereas GPUs want large matmuls with high arithmetic intensity. Ideally, we'd like to keep the favorable scaling of linear attention while restructuring the computation into something the hardware can actually make good use of.

Fortunately, there's something useful about our recurrence that we haven't taken advantage of yet. Recall that the state is simply a running sum of outer products:

$$
S_t = X_1+\cdots+X_t,\qquad X_t=v_tk_t^\top
$$

Each token contributes an outer product, and we accumulate these contributions into the state. But since we're just adding matrices together, there's no fundamental reason we have to do this one token at a time. We could group several tokens together and compute their combined contribution in one matrix multiplication.

This is the idea behind **chunkwise parallel linear attention**.

Instead of processing the entire sequence recurrently, we split it into chunks of $C$ tokens. We then maintain the recurrent state between chunks, while using parallel matrix multiplications within each chunk. This gives us a middle ground between the two approaches we've considered: we avoid the long chain of tiny operations from naive recurrence, without falling back to an $L\times L$ attention matrix.

More precisely, we reduce the sequential dependency chain from $L$ token-level steps to roughly $L/C$ chunk-level steps, while replacing many of the small operations with dense matmuls that can take advantage of GPU tensor cores. For vanilla linear attention, the additive state also admits parallel prefix scans, so chunking isn't the only way to parallelize the computation. Its appeal is that it gives us a particularly hardware-friendly way of doing so.

Let's work through this process in more detail shall we. There are roughly $L/C$ chunks. We'll call the state at the **start** of chunk $i$ $S_{[i]}$, and use $Q_{[i]},K_{[i]},V_{[i]}\in\mathbb{R}^{C\times d}$ for its token representations, now stored as rows.

#### step 1: update the state between chunks

Rather than applying $C$ outer-product updates individually, we can accumulate all of them in one matmul:

$$
\boxed{S_{[i+1]}=S_{[i]}+V_{[i]}^\top K_{[i]}}
$$

This is exactly the same update as before, just grouped together, because

$$
V_{[i]}^\top K_{[i]}
=\sum_{t\in\text{chunk }i}v_tk_t^\top
$$

The cost for one chunk is $O(Cd^2)$. With $L/C$ chunks, the total cost of updating the boundary states is $O(Ld^2)$. So we haven't changed the arithmetic order at all. We've just replaced many small outer products with much larger, more GPU-friendly matmuls.

#### step 2: compute the outputs inside a chunk

Now consider a query in the current chunk. It needs to attend to two sets of tokens: everything in previous chunks, and the earlier tokens (including itself) in the current chunk.

Those two parts can be computed separately. All previous chunks have already been summarized into $S_{[i]}$. Their contribution is simply

$$
Q_{[i]}S_{[i]}^\top
$$

which costs $O(Cd^2)$ per chunk. For tokens within the current chunk, we can use ordinary causal token-token interactions, but the attention matrix is only $C\times C$ rather than $L\times L$:

$$
(Q_{[i]}K_{[i]}^\top\odot M_C)V_{[i]}
$$

Here $M_C$ is a binary causal mask, and $\odot$ denotes elementwise multiplication. This part costs $O(C^2d)$ per chunk.

Putting both contributions together,

$$
\boxed{
O_{[i]}=Q_{[i]}S_{[i]}^\top
+ (Q_{[i]}K_{[i]}^\top\odot M_C)V_{[i]}
}
$$

I think this equation captures the whole algorithm nicely. The first term reads the compressed memory of all earlier chunks. The second handles the token-level causal interactions that haven't yet been folded into that memory.

Summing the costs over all chunks gives

$$
\underbrace{O(Ld^2)}_{\text{state updates and historical reads}}
+
\underbrace{O(LCd)}_{\text{within-chunk attention}}
$$

Compare this with $O(L^2d)$ for full attention. If $C$ is kept much smaller than $L$, we retain the favorable sequence-length scaling while exposing dense matmuls inside each block.

The limiting cases help us understand the trade-off: $C=1$ gets us back to the token-wise recurrent form, $C=L$ gets us the full parallel, quadratic causal form. Intermediate chunk sizes, often around 64, 128, or 256 in practical implementations, trade a shorter cross-chunk dependency chain against more work within each chunk.

Notice that chunking doesn't eliminate recurrence. The state still moves from chunk to chunk. We've reduced the naive sequential chain from $L$ token steps to roughly $L/C$ chunk steps, and made the work at each step much more suitable for GPU hardware.

## mental model

Softmax attention:

```text
all tokens x all tokens
good GPU shape, but quadratic in sequence length
```

Recurrent linear attention:

```text
token -> token -> token -> token
linear in sequence length, great for decoding, poor GPU utilization for naive training
```

Chunkwise linear attention:

```text
chunk -> chunk -> chunk
inside each chunk: parallel dense matmuls
between chunks: compact recurrent state
```

## hardware performance
Songlin comments on the speedup achieved after implementing both the recurrent and chunkwise version in Triton:

<img src="/images/chunkwiseparallelform.png" alt="chunkwise parallel form vs recurrent baseline" style="width: 70%; height: auto; display: block; margin: 2rem auto;" />

> *chunkwise parallel approach consistently outperforms the recurrent baseline...this performance advantage grows more pronounced under two key conditions: as sequences get longer and as head dimensions increase.*

Why? Songlin points to two issues (1) existing recurrent implementations process token-by-token, relying instead on both batch and head dimension to saturate GPU cores:

> *While this strategy worked well with moderate sequence lengths and larger batch sizes, it faces challenges in modern training scenarios. Today's models increasingly work with longer sequences or larger model parameters, often necessitating smaller batch sizes for memory efficiency. This shift was notably highlighted in the FlashAttention2 paper, which identified sequence-level parallelism as crucial for training.*

when `batch size x head dim` is small, there isn't enough parallel work to keep modern GPUs fully utilized.

(2):

> *Tensor cores are designed around matmuls offering up to >16x speedup compared to other operations with equivalent FLOP counts. Recurrent implementations, despite requiring fewer total FLOPs, struggle to effectively leverage these hardware accelerators...Our chunkwise implementation, in contrast, restructures the computation to maximize use of tensor cores, achieving better real-world performance despite higher theoretical FLOP counts. This performance analysis illustrates a crucial principle in modern hardware-efficient deep learning: raw FLOP counts don't always translate directly to wall-clock time. The ability to leverage specialized hardware accelerators and maintain high GPU utilization often matters more than theoretical operation counts.*

## the catch: a fixed-size memory

We've addressed the computational side. But there's a more fundamental limitation of vanilla linear attention, one which the associative-memory perspective makes quite clear. Recall that our state is just

$$
S=\sum_i v_ik_i^\top
$$

If we try to retrieve the value associated with some key $k_j$, we get

$$
Sk_j=\sum_i v_i(k_i^\top k_j)
$$

Let's assume the keys are normalized. If they were also perfectly orthogonal, then $k_i^\top k_j=0$ whenever $i\neq j$, while $k_j^\top k_j=1$. We'd retrieve $v_j$ almost perfectly. But real keys aren't orthogonal, and there's only so much room in a $d$-dimensional key space. As we add more associations, the other terms don't disappear:

$$
Sk_j=
\underbrace{v_j}_{\text{desired value}}
+
\underbrace{\sum_{i\neq j}v_i(k_i^\top k_j)}_{\text{interference}}
$$

leading to a memory-overload problem. Multiple keys overlap, so retrieving one association also picks up contributions from others. Naturally, compressing an arbitrarily long history into a fixed-size state necessarily limits what we can preserve. So we want to build the memory with care. So what does vanilla linear attention do, well its update function is append-only...

$$
S_t=S_{t-1}+v_tk_t^\top
$$

Every new token adds another association, even if its key is very similar to something we've already stored. There's no notion of removing stale information or correcting an existing value. The memory simply keeps accumulating writes.

## first, give the memory a way to forget

Probably the simplest idea is to decay the old state before adding a new association. Language modeling has a strong recency bias anyway, so it's reasonable to expect that more recent information will often matter more.

We can start with a fixed decay factor:

$$
\boxed{S_t=\gamma S_{t-1}+v_tk_t^\top},\qquad 0<\gamma<1
$$

Expanding the recurrence shows what this does:

$$
S_t=\sum_{i\leq t}\gamma^{t-i}v_ik_i^\top
$$

Older associations are exponentially downweighted. A similar decay mechanic was used by MiniMax last year in their Lightning Attention. The obvious limitation is that the same forgetting rate applies regardless of what we're processing. Sometimes old information should be discarded quickly; sometimes we'd like to retain it for a long time.

So let's make the decay **data dependent**:

$$
\boxed{S_t=\gamma_tS_{t-1}+v_tk_t^\top}
$$

Now $\gamma_t$ can be predicted from the current token representation, much like we predict $q_t$, $k_t$, and $v_t$. Instead of having one fixed timescale, the model can decide how much of its previous state to retain at each step. This broader style of gated state decay appears in architectures such as Mamba-2 and mLSTM.

We can easily take this one step further. Why should every part of the state be forgotten at the same rate? In a matrix-valued memory, we could apply more fine-grained decay:

$$
\boxed{S_t=G_t\odot S_{t-1}+v_tk_t^\top}
$$

where $G_t$ is a gate with the same shape as $S_t$. Gated Linear Attention (GLA) explores this direction, using structured gates rather than an arbitrary dense matrix of unrelated decay factors.

One must be careful however. That structural constraint matters. A completely unrestricted elementwise gate would make it difficult to retain the efficient chunkwise computation we just spent so much time developing. We want more expressive memory updates, but we still need to be able to train them efficiently.

We're definately improving on the original design, we're now able to decay parts of the state in a data-dependent way. But we're still just adding to whatever remains of the memory. What if a key has already exists in memory and we'd like to *change* its value rather than pile another value on top?


## DeltaNet: from adding memories to editing them

Suppose we're about to write a new key-value association $(k_t,v_t)$. Before writing, why not ask what our memory already returns for that key?

$$
v_{\mathrm{old}}=S_{t-1}k_t
$$

We want the memory to return $v_t$, but right now it returns $v_{\mathrm{old}}$. Rather than adding the full value again, we could write only the difference:

$$
\boxed{
S_t=S_{t-1}
+\beta_t(v_t-S_{t-1}k_t)k_t^\top
}
$$

This is the **delta rule**. The term in parentheses is the retrieval error, and $\beta_t$ controls how strongly we correct it. Compare it with vanilla linear attention:

$$
\begin{aligned}
\text{linear attention:}\quad
&S_t=S_{t-1}+v_tk_t^\top\\
\text{DeltaNet:}\quad
&S_t=S_{t-1}+\beta_t(v_t-v_{\mathrm{old}})k_t^\top
\end{aligned}
$$

The difference is small algebraically but coneptually very important. Vanilla linear attention writes without checking what's already there. DeltaNet uses the current memory to determine what needs to change. 

We can see the overwrite behavior more explicitly. Assuming $\|k_t\|^2=1$, reading back the same key after the update gives

$$
S_tk_t=(1-\beta_t)v_{\mathrm{old}}+\beta_t v_t
$$

So $\beta_t=0$ preserves the old retrieved value, $\beta_t=1$ replaces it with the new target along that key direction, and values in between interpolate between the two.

Equivalently, we can expand the update into an erase-and-write form:

$$
\boxed{
S_t=S_{t-1}(I-\beta_t k_tk_t^\top)
+\beta_t v_tk_t^\top
}
$$

The first term adjusts what the old memory says along $k_t$; the second writes the new association. Obviously the state size hasn't changed, we're just giving the model a much more useful way to manage its capacity. This doesn't magically eliminate all interference between correlated keys. Editing one direction in a finite-dimensional memory can still affect others. But it is a meaningful step beyond an append-only sum of outer products.

## Gated DeltaNet

At this point we have two useful mechanisms. Gated linear attention can forget old state, while DeltaNet can correct an existing association. They address slightly different problems, so it's natural to combine them.

Start by decaying the previous memory with a data-dependent scalar gate $\alpha_t$:

$$
\bar S_{t-1}=\alpha_tS_{t-1}
$$

Then perform the delta-rule update against that *decayed* memory:

$$
\boxed{
S_t=\bar S_{t-1}
+\beta_t(v_t-\bar S_{t-1}k_t)k_t^\top
}
$$

Substituting $\bar S_{t-1}=\alpha_tS_{t-1}$ gives

$$
\boxed{
S_t=\alpha_tS_{t-1}(I-\beta_tk_tk_t^\top)
+\beta_tv_tk_t^\top
}
$$

This is Gated DeltaNet. The roles of the two gates are different: $\alpha_t$ decides how much of the old memory survives, while $\beta_t$ decides how strongly we update the association at the current key.

One detail worth noticing is the order of operations. The prediction used by the delta rule is now

$$
v_{\mathrm{old}}=\alpha_tS_{t-1}k_t
$$

not $S_{t-1}k_t$. We're correcting what the memory returns **after** forgetting, rather than what it returned before.

It's always nice when things make sense intuitively. This combination means we can make room by decaying old information, and we can use that room more carefully by writing prediction errors rather than unconditional additions. I feel like the next step should almost feel routine at this point, we've combined the two mechanisms together, but now someone realizes that the forgetting factor could be even more fine-grained. Right now we've got one scalar gate per attention head, why stop there?

## Kimi Delta Attention

Kimi Delta Attention (KDA), used in Kimi Linear and Kimi K3, takes Gated DeltaNet and makes the decay more fine-grained. The succession is very simple, GDN uses one forget gate per head. KDA uses one forget gate per channel within a head. 

It's worth making the multi-head setup explicit here. Each head gets its own projected queries, keys, and values, and maintains its own recurrent memory. With key/query dimension $d_k$ and value dimension $d_v$, we've been using the convention

$$
q_t,k_t\in\mathbb{R}^{d_k},\quad
v_t\in\mathbb{R}^{d_v},\quad
S_t\in\mathbb{R}^{d_v\times d_k}
$$

so the output is $o_t=S_tq_t$. With $H$ heads, the total recurrent-state size per layer is $Hd_vd_k$, not a single $d_{\mathrm{model}}\times d_{\mathrm{model}}$ matrix. (KDA is also commonly written with the state transposed; I'm keeping our original convention to avoid switching orientations halfway through.)

In Gated DeltaNet, $\alpha_t$ is a scalar, so every entry of a head's memory gets multiplied by the same number. In KDA, it becomes a vector over key channels:

$$
\alpha_t\in(0,1)^{d_k}
$$

We can express the corresponding decay as a diagonal matrix. First, decay the old memory:

$$
\boxed{
\bar S_{t-1}=S_{t-1}\operatorname{Diag}(\alpha_t)
}
$$

The multiplication is on the right because, with our convention, the key dimension indexes the columns of $S_t$. Each key channel can therefore retain or forget information at its own rate. Then apply the same delta-rule correction as before:

$$
\boxed{
S_t=\bar S_{t-1}
+\beta_t(v_t-\bar S_{t-1}k_t)k_t^\top
}
$$

Expanding both steps gives the full recurrence:

$$
\boxed{
S_t=S_{t-1}\operatorname{Diag}(\alpha_t)
(I-\beta_tk_tk_t^\top)
+\beta_tv_tk_t^\top
}
$$

The two gates still do what they did in Gated DeltaNet. $\beta_t$ controls the correction at the current key. But $\alpha_t$ can now preserve some key channels while aggressively decaying others, rather than treating the whole head as one unit.