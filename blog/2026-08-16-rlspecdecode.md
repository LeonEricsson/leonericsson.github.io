---
layout: post
title: speculative decoding in rl
year: 2026
type: blog
---

**pre-reading expectations. what am i looking for?**

post training recipe - opd? mopd? are we RLing distinct agents, if so how are we combining them?
attention variant: linear hybrid or sparse route?
rl algorithm - numerical instability? quantized rl?
rl environments - any details on data pipeline (likely not)?

### Kimi Delta Attention

Kimi Delta Attention (KDA), used in Kimi Linear and Kimi K3, is best understood as a refinement of Gated DeltaNet.

The main conceptual change is:

```text
Gated DeltaNet: one forget gate per head
KDA: one forget gate per key channel inside each head
```

Before writing the KDA update, it is useful to make the multi-head structure explicit. Linear attention variants still fit into the usual multi-head attention architecture:

```text
x_t
 -> project to q, k, v for each head
 -> each head runs its own recurrent state update
 -> concatenate head outputs
 -> output projection
```

For one head:

$$
q_t,k_t \in \mathbb{R}^{d_k},\qquad v_t \in \mathbb{R}^{d_v}
$$

and the recurrent state is:

$$
S_t \in \mathbb{R}^{d_k \times d_v}
$$

This follows the Kimi paper convention, where the output is read as:

$$
o_t = S_t^\top q_t
$$

So each head has its own associative memory. If there are $H$ heads, the total state per layer is:

$$
H \times d_k \times d_v
$$

not:

$$
d_{\text{model}} \times d_{\text{model}}
$$

This is important because it keeps the recurrent state manageable.

In Gated DeltaNet, the forget gate is a scalar per head:

$$
\alpha_t \in (0,1)
$$

and the update is:

$$
S_t =
\alpha_t(I - \beta_t k_tk_t^\top)S_{t-1}
+
\beta_t k_tv_t^\top
$$

This means the whole memory matrix for that head is decayed by the same amount.

KDA changes $\alpha_t$ from a scalar into a vector:

$$
\alpha_t \in (0,1)^{d_k}
$$

and applies it as a diagonal matrix over the key/address dimension:

$$
\mathrm{Diag}(\alpha_t) \in \mathbb{R}^{d_k \times d_k}
$$

making KDA's recurrent form:

$$
S_t =
(I - \beta_t k_tk_t^\top)\mathrm{Diag}(\alpha_t)S_{t-1}
+
\beta_t k_tv_t^\top
$$

The clean way to read this is in two stages.

**Decay Old Memory**

$$
\bar{S}_{t-1} = \mathrm{Diag}(\alpha_t)S_{t-1}
$$

This scales the rows of $S_{t-1}$, meaning each key/address channel gets its own forgetting rate:

```text
key channel 1: decay by alpha_1
key channel 2: decay by alpha_2
key channel 3: decay by alpha_3
...
```

**Apply Delta-Rule Update**

Then apply the usual delta-rule correction to the decayed memory:

$$
S_t =
\bar{S}_{t-1}
+
\beta_t k_t(v_t - \bar{S}_{t-1}^\top k_t)^\top
$$

So the prediction is not made from the raw old memory. It is made from the already-forgotten memory:

$$
v_{\text{old}} = \bar{S}_{t-1}^\top k_t
$$

Then KDA writes the error correction:

$$
\beta_t k_t(v_t - v_{\text{old}})^\top
$$

This gives $\alpha_t$ and $\beta_t$ separate roles:

- $\alpha_t$: what old memory survives into this step
- $\beta_t$: how strongly the new association is written

### Attention Residuals

*todo*

### MLA

MLA is easiest to remember as the same dense attention written in two forms:

```text
training / prefill: MHA with small per-head K/V
decode:            MQA over one wide shared KV latent
```

It does not try to reduce the quadratic attention cost during training/prefill. The win is that K/V are parameterised through a compressed latent, letting decode avoid storing expanded per-head K/V.

For K3:

- $d_c = 512$: compressed KV latent
- $d_{\text{NoPE}} = 128$: per-head content Q/K dimension
- $d_R = 64$: legacy RoPE branch
- $d_v = 128$
- $H$: query heads
##### Training / Prefill

The KV down-projection produces
$$
W_{DKV}x_t = [c_t;s_t],
\qquad
c_t\in\mathbb{R}^{512},
\quad
s_t\in\mathbb{R}^{64}.
$$

Only $c_t$ goes through the expensive per-head up-projection:

$$
W_{UK}:\mathbb{R}^{512}\rightarrow\mathbb{R}^{H\cdot128},
\qquad
W_{UV}:\mathbb{R}^{512}\rightarrow\mathbb{R}^{H\cdot128}.
$$

So
$$
c_t\rightarrow\{k^{128}_{t,h},v^{128}_{t,h}\}_{h=1}^{H}.
$$
The 64-d branch bypasses this and is shared across heads:
$$
k_{t,h}=[k^{128}_{t,h};s_t^{64}],
\qquad
q_{t,h}=[q^{128}_{t,h};q^{64}_{t,h}].
$$
Thus training/prefill is ordinary dense MHA with
$$
\boxed{d_{qk}=128+64=192,\qquad d_v=128.}
$$

##### Decode - absorbing the up-projection aka "MLA Trick"

For the content branch,
$$
k^{128}_{i,h}=W_{UK,h}c_i
$$
and therefore

$$
(q^{128}_{t,h})^\top W_{UK,h}c_i
=
(W_{UK,h}^\top q^{128}_{t,h})^\top c_i.
$$
Define
$$
\tilde q_{t,h}=W_{UK,h}^\top q^{128}_{t,h}
\in\mathbb{R}^{512}.
$$
Now every query head attends directly to the same cached $c_i\in\mathbb{R}^{512}$. This is the MQA view: many distinct query heads, one shared KV representation. $c_i$ is broadcast logically, not duplicated in the cache.

The 64-d branch was never inside $W_{UK}$, so it remains alongside the absorbed content branch for the QK operation:

$$
[\tilde q^{512}_{t,h};q^{64}_{t,h}]^\top
[c_i^{512};s_i^{64}].
$$
Hence decode is effectively a 576 head_dim MQA
$$
\boxed{d_{qk}=512+64=576,\qquad d_v=512}
$$

**Mental model:** training sees MHA-192/128; decode rearranges the exact same computation into MQA-576/512.
##### attention head dimensions 

Jianlin has repeatedly found that, for a fixed KV-cache budget, spending dimensions on a **wider single KV head** works better than splitting the same budget across more KV heads/groups. For one head,

$$
L_h=Q_hK_h^\top,
\qquad
\mathrm{rank}(L_h)\le d_k.
$$

So $d_k$ is a real capacity axis: more heads gives more separate attention maps, while larger head dimension makes each map's matching space richer. This is easy to miss in MLA because training visibly uses 128-d content heads, but absorption exposes a much wider 512-d latent matching space at decode. The empirical rule Jianlin arrives at is roughly:

> For a fixed KV-cache size $C$, prefer one MQA-style KV representation with head dimension about $C$, shared by K and V.

Rather than divide $C$ cached numbers across several KV heads and separate K/V spaces, use one shared latent $c_i\in\mathbb{R}^{C}.$ MLA's $c_i\in\mathbb{R}^{512}$ is close to this ideal. The absorbed 512-d query is still generated from the original 128-d query, so it is low-rank constrained rather than a completely free native 512-d query.
##### Why does K3 retain RoPE split despite removing RoPE encoding?

Traditional MLA needed the separate 64-d branch because RoPE is position-dependent and cannot participate in the same absorption as the NoPE content path.

K3 removes RoPE from its MLA layers, but keeps the old structural split:

$$
x_t\xrightarrow{W_{DKV}}[c_t^{512};s_t^{64}].
$$

The 64 dimensions are now just learned NoPE features. They still bypass the KV up-projection and are concatenated directly into Q/K. A cleaner NoPE design would make the whole thing one latent:

$$
x_t\rightarrow c_t^{576}
\rightarrow
\{k^{192}_{t,h},v^{128}_{t,h}\}_{h=1}^{H}.
$$
But this enlarges the expensive per-head up-projection:

```text
current K3:   512 -> H * (128 + 128)
unified path: 576 -> H * (192 + 128)
```

That extra compute did not buy enough performance to justify it. Keeping the old split also preserves MLA infrastructure. Simply deleting the 64-d branch would be cheaper, but would reduce effective QK head dimension from $576$ to $512$, which runs against the finding that wider head dimensions matter. So the retained branch is best viewed as **cheap extra attention-score capacity**:

$$
\ell_{tih}
=
(\tilde q^{512}_{t,h})^\top c_i^{512}
+
(q^{64}_{t,h})^\top s_i^{64}.
$$
It enlarges the QK matching space by 64 dimensions without carrying those dimensions through the full per-head K/V projection machinery.

**thoughts:** extra dimensions seem especially valuable when they enlarge the attention matching space itself; making those same dimensions participate in the full latent-to-per-head projections is much less cost-effective. given that RoPE information is no longer carried in the separate path, we now have much more flexibility in changing the relationship between these two paths. it seems worth exploring what the actual split should be, why 512;64? 
### Stable LatentMoe

Let

- $d$: model hidden dimension
- $\ell < d$: latent expert dimension
- $N$: total experts
- $K$: active experts per token
- $\alpha = d/\ell$: compression factor

##### MoE

For $x \in \mathbb{R}^d$:

$$
x
\xrightarrow{\text{top-}K}
E_{i_1}(x), \ldots, E_{i_K}(x)
\xrightarrow{\text{combine}}
y \in \mathbb{R}^d.
$$

Experts operate directly in the $d$-dimensional space. Roughly:

$$
\text{total parameters} \propto Nd,
\qquad
\text{compute / memory traffic} \propto Kd,
\qquad
\text{all-to-all} \propto Kd.
$$

##### LatentMoE

First project into a smaller latent space:

$$
x \in \mathbb{R}^d
\rightarrow
z \in \mathbb{R}^{\ell}
\xrightarrow{\text{top-}K}
E_{i_1}(z), \ldots, E_{i_K}(z)
\rightarrow
y \in \mathbb{R}^d.
$$

With

$$
\ell = \frac{d}{\alpha},
$$

each expert and each routed activation becomes $\alpha$ times cheaper, which we instead spend on scaling the number of total and active experts:

$$
\boxed{
N' = \alpha N,
\qquad
K' = \alpha K
}
$$

Then the budgets transfer cleanly:

$$
N'\ell
=
(\alpha N)\frac{d}{\alpha}
=
Nd
$$

**total parameters stay fixed**

$$
K'\ell
=
(\alpha K)\frac{d}{\alpha}
=
Kd
$$

**compute, expert-weight memory traffic, and all-to-all communication also stay roughly fixed**.

We trade expert width for more experts, which has the same training and inference costs but with better performance.

##### Kimi K3's Stable LatentMoe

Employs LatentMoe with $\alpha = 2$, meaning $d = 7167, \ell = 3584$ and moving from K2's 384 select 8 experts to 896 select 16 experts. So they halve routed width and exactly double active experts. To first order, the routed all-to-all volume $K\ell$ has therefore not decreased at all.

The "Stable" label comes from adding a RMS Norm prior to the up-projection (decompression) to stabilise training
![alt text|468](image%2014.png)

---

### Stable LatentMoE

Let

* $d$: model hidden dimension
* $\ell < d$: latent expert dimension
* $m$: expert intermediate dimension
* $N$: total experts
* $K$: active experts per token
* $E$: expert-parallel ranks
* $t$: tokens across the EP domain
* $\alpha=d/\ell$: latent compression factor

##### The MoE systems picture

For a SwiGLU expert, ignoring datatype constants:

$$
\text{expert params}\propto Ndm,
\qquad
\text{active FLOPs}\propto Kdm,
$$

while latency-critical expert-weight traffic is roughly

$$
\text{HBM traffic}\propto Kdm,
$$

and routed all-to-all traffic is

$$
\text{network traffic}\propto Kd.
$$

There are two important operating regimes.

**Latency-critical / small batch:** expert GEMMs have too few tokens to amortize loading their weights and are HBM-bandwidth bound.

**Throughput / large batch:** expert GEMMs have enough tokens to reach the compute roofline. Now distributed MoE performance is governed by the relationship between expert compute and all-to-all communication.

For $t$ tokens uniformly routed across the EP domain:

$$
T_{\text{comm}}
\approx
\frac{2tKd}{EB_{\text{net}}},
$$

where the factor $2$ is dispatch + combine, while

$$
T_{\text{compute}}
\approx
\frac{6tKdm}{EF}
$$

for SwiGLU's gate, up and down projections.

Thus

$$
\frac{T_{\text{comm}}}{T_{\text{compute}}}
\approx
\frac{F}{3mB_{\text{net}}}.
$$

Once the expert GEMMs are compute-bound, $t,K,d,N,E$ all disappear from this ratio. Changing $d$ changes the **absolute** compute and communication costs together, but not their balance.

The important prerequisite is that each expert receives enough tokens:

$$
t_{\exp}=\frac{tK}{N}.
$$

If $t_{\exp}$ is too small, the expert GEMM is HBM-bound and the compute-vs-network roofline above is not the right model. This is why both latency-critical and throughput-oriented regimes have to be considered separately.

##### Cost-model lens

Architectural changes should be evaluated mechanically across the relevant serving regimes.

**Latency-critical / small batch:** expert execution is typically bandwidth-bound. Speed is largely about reducing **bytes moved**:

* expert weights / activations through HBM
* routed activations through the network

**Throughput / large batch:** first ask whether the expert GEMMs have enough arithmetic intensity to be **compute-bound rather than HBM-bound**. If they do, then ask whether there is enough expert compute to hide the all-to-all communication.

So for a change such as

$$
d\rightarrow\ell,
$$

don't reason about "cost" as one quantity. Ask separately:

$$
\boxed{
\text{HBM bytes},\quad
\text{network bytes},\quad
\text{FLOPs},\quad
\text{arithmetic intensity},\quad
\text{comm/compute ratio}
}
$$

A change can improve absolute cost without changing a roofline ratio, or preserve asymptotic cost while changing whether the underlying kernels actually reach that roofline.

##### Why reduce $d$?

To improve inference efficiency we want to reduce HBM traffic and all-to-all traffic.

* memory cost scales with $d$ and $m$
* communication scales with $K$ and $d$
* reducing $K$ or $m$ reduces the expert's nonlinear capacity and tends to hurt quality
* increasing $N$ and $K$ tends to improve expert specialization / mixture diversity

This leaves $d$ as the attractive dimension to compress. LatentMoE therefore projects

$$
x\in\mathbb R^d
\rightarrow
z\in\mathbb R^\ell,
\qquad
\ell=\frac d\alpha,
$$

and performs routing, communication and expert computation in this smaller space.

Each routed expert invocation is now approximately $\alpha$ times cheaper. LatentMoE proposes two variants of what to do with these cost savings.

##### $\ell$-MoE$_{\text{eff}}$: bank the savings

Scale the expert pool but keep the number of active experts fixed:

$$
\boxed{
\ell=\frac d\alpha,\qquad
N'=\alpha N,\qquad
K'=K
}
$$

Total routed expert parameters stay fixed:

$$
N'\ell m = 
 (\alpha N)\frac d\alpha m

= 
Ndm.
$$

But active expert cost falls:

$$
K'\ell m
=

\frac{Kdm}{\alpha},
$$

and communication falls by the same factor:

$$
K'\ell
=

\frac{Kd}{\alpha}.
$$

So:

$$
\boxed{\text{MoE}_{\text{eff}}:\quad \text{retain accuracy, improve inference efficiency}}
$$

The larger $N$ spends the reduced **per-expert parameter cost** on a richer expert pool, while the savings from the smaller routed representation are banked as lower active cost.

##### $\ell$-MoE$_{\text{acc}}$: reinvest the savings

Instead also scale the number of active experts:

$$
\boxed{
\ell=\frac d\alpha,\qquad
N'=\alpha N,\qquad
K'=\alpha K
}
$$

Total parameters again remain fixed:

$$
N'\ell m=Ndm.
$$

Now active expert cost is also restored to baseline:

$$
K'\ell m
=

 (\alpha K)\frac d\alpha m =

Kdm,
$$

and so is all-to-all traffic:

$$
K'\ell
=

 (\alpha K)\frac d\alpha =

Kd.
$$

Hence compute, active-weight traffic and communication remain approximately iso-cost, but each token gets more active experts from a larger expert pool:

$$
\boxed{\text{MoE}_{\text{acc}}:\quad \text{same inference efficiency, better accuracy}}
$$

This is the main LatentMoE result: **compress expert representation width, then spend that efficiency on expert diversity rather than keeping wider experts.**

##### Kimi K3's Stable LatentMoE

K3 uses $\alpha=2$:

$$
d=7168,\qquad \ell=3584,
$$

and moves from K2's select-8 routing to select-16:

$$
8\cdot7168
==========

16\cdot3584.
$$

So K3 spends the latent-width saving on twice as many active experts: its routed active width, and therefore first-order expert compute / all-to-all cost, stays roughly unchanged. It also expands the expert pool from $384$ to $896$.

The **Stable** modification adds RMSNorm before the latent output is projected back into the model hidden dimension.
