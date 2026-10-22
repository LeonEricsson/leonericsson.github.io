---
layout: post
title: speculative decoding in rl
year: 2026
type: blog
---

Rollout is the dominant cost of RL. Efficiency improvements targeting rollout therefore provide high return on investment. Speculative decoding is a very common inference optimization, widely used when serving models. RL obviously runs a lot of inference, so are we running spec decode under RL, if not, why? I'd like to go through some reports/papers of applying spec decode to RL to get an understanding of the state of spec decode in rl. 

starting with a brief primer on speculative decoding. For a target policy $p$, speculative decoding introduces a cheaper drafter $q$ which proposes $\gamma$ tokens sequentially. The target verifies the entire block in parallel. If the rejection-sampling correction is exact, the resulting sequence is still sampled from $p$; the drafter changes *how* we obtain the sample, not its distribution.

For target-token decode cost $T_p$, drafter-token cost $T_q$, verification cost $T_V$, and block efficiency $\tau$:

$$
T_{\mathrm{SD}} = \gamma T_q + T_V
$$

$$
\text{Speedup} =
\tau \frac{T_p}{\gamma T_q + T_V}.
$$

Here $\tau$ is roughly the number of target-distributed tokens emitted per draft/verify iteration: accepted draft tokens plus the final target-sampled token. Its maximum is $\gamma+1$.

This gives two fundamental requirements for useful SD:

1. **The drafter must be good enough.** High agreement between $q$ and $p$ gives large $\tau$.
2. **The system must be in a regime where drafting + verification is cheaper than ordinary decoding.**

The second point is easy to miss. Parallel verification is attractive when normal decoding is memory-bandwidth bound and leaves arithmetic throughput unused. If the target is already compute-bound, verifying several tokens simultaneously consumes real additional compute and SD can lose even with a strong drafter.

### RL introduces a moving target

Serving usually assumes a fixed target model. RL does not:

$$
p_{\theta_0}
\rightarrow
p_{\theta_1}
\rightarrow
p_{\theta_2}
\rightarrow \cdots
$$

A learned drafter trained to approximate $p_{\theta_0}$ can therefore become stale:

$$
q_{\phi_0} \approx p_{\theta_0},
\qquad
q_{\phi_0} \not\approx p_{\theta_t}.
$$

As target/drafter mismatch increases, acceptance falls and block efficiency decreases. A frozen EAGLE-3 drafter, for example, shows acceptance length falling substantially over roughly 100 RL updates as the actor evolves.

This creates a general design problem:

$$
\boxed{\text{How do we keep }q\text{ aligned with an evolving }p?}
$$

Broadly there are two approaches:

* continually adapt a lightweight drafter to the current policy;
* construct the drafter directly from the current policy so explicit adaptation is unnecessary.

### SD efficiency depends on the runtime regime

SD speedup depends strongly on active batch size.

At large batch sizes, ordinary decoding can already saturate the GPU. There is little unused compute for speculative verification to exploit, while SD still pays drafting and synchronization costs. In one measured setup, the same SD configuration gives $1.46\times$ speedup at batch size 2 but only $0.76\times$ at batch size 32.

This means the optimal decoding strategy can change during generation:

$$
B \uparrow
\Rightarrow
\text{AR often preferable}
$$

$$
B \downarrow
\Rightarrow
\text{SD increasingly attractive}.
$$

This matters especially in **synchronous RL**. A synchronous iteration roughly looks like

$$
\text{generate finite rollout batch under }\theta_t
\rightarrow
\text{wait for completion}
\rightarrow
\text{update actor}
\rightarrow
\theta_{t+1}.
$$

Responses have different lengths, so short responses finish first and the active batch drains:

$$
B_{\text{large}}
\rightarrow
B_{\text{medium}}
\rightarrow
B_{\text{small}}.
$$

Eventually a few long generations determine the makespan while the inference hardware becomes underutilized. This creates a natural opportunity to begin with AR and enable SD only after entering the low-batch tail.

This argument does **not** transfer directly to modern asynchronous RL systems. An asynchronous inference engine continously enques new rollouts, keeping accelerators topped up.

### the optimal SD configuration changes over time

Even after solving drafter staleness, a fixed SD configuration need not remain optimal.

Draft length, tree width, speculative rounds, and other SD parameters trade additional drafting work against the probability of amortizing that work through successful acceptance. Their optimum depends on both **drafter quality** and **system state**.

In RL, both can change:

$$
q_t \text{ quality changes}
\qquad\text{and}\qquad
B_t,\ S_t \text{ change}.
$$

For example, a drafter that becomes more accurate can profitably increase its draft length:

$$
\tau \uparrow
\Rightarrow
\gamma_{\mathrm{optimal}} \uparrow.
$$

The general lesson is that **SD configuration should be treated as a runtime control problem rather than a static inference hyperparameter.**

---

### EfficientRollout

EfficientRollout removes the learned-drafter maintenance problem by constructing the drafter directly from the current actor.

At each RL step:

$$
q_t = Q(p_{\theta_t}),
$$

where $Q$ is 4-bit weight quantization. The drafter therefore has essentially the same architecture as the target, but heavily quantized weights. Since it is regenerated from the current policy, it cannot become stale relative to an old actor snapshot.

**Why weight quantization?**

In the small-batch rollout tail, over $90%$ of decode latency comes from dense projections such as FFN, QKVO and the LM head, while attention is comparatively cheap.

This regime is largely dominated by repeatedly loading model parameters from memory. Roughly,

$$
T_p \sim \frac{W_{\mathrm{target}}}{BW}.
$$

Representing the dominant projection weights in W4 therefore substantially lowers weight traffic and makes the drafter cheaper. This is why weight quantization is more useful here than something like sparse attention, which attacks a relatively small fraction of latency.

The tradeoff is unusual:

$$
\boxed{
\text{very large drafter}
+
\text{extremely high agreement}
+
\text{no drafter training}
}
$$

The quantized drafter achieves roughly $96-98\%$ token acceptance. EfficientRollout uses draft lengths from

$$
\Gamma={5,7,9,11},
$$

and on Qwen2.5-7B moves from $\gamma \approx 5$ early toward $\gamma \approx 11$ later as block efficiency improves.

**Regime-aware toggle.** EfficientRollout begins rollout decoding with AR and uses a calibrated roofline model to predict whether

$$
\frac{\tau T_p}
{\gamma T_q + T_V}
$$

is favorable under the current batch size, sequence length, block efficiency and draft length. Once predicted speedup passes a safety margin, it switches

$$
AR \rightarrow SD.
$$

Because its synchronous active batch only decreases, this is a one-way switch.

**Adaptive draft length.** Measured $\tau$ is used as feedback. Persistently high utilization of the current block increases $\gamma$; poor utilization decreases it.

The method therefore amounts to:

- quantize current actor into drafter
- begin rollout with AR
- switch to SD when the system enters a favorable regime
- adapt $\gamma$ as drafter quality changes.

Despite excellent acceptance, the realized gains are only around $10-20%$ in rollout latency and $8-13%$ end-to-end because the drafter is still a full target-shaped network.

A useful characterization of the method is

$$
\boxed{\text{excellent speculation quality, expensive speculation}.}
$$

### ReSpec

ReSpec takes the opposite approach: use a **small learned EAGLE-3 drafter**, accept that it will become stale as the actor changes, and continuously adapt it during RL.

The drafter is initially pretrained offline. During RL, speculative decoding itself becomes the source of online training data. For each speculative block, the drafter proposes tokens and the target verifies them. Verification already computes the target distribution on those positions, so the expensive teacher-side supervision is produced **as a by-product of generation**:

$$
q_{\phi_t}\text{ drafts}
\longrightarrow
p_{\theta_t}\text{ verifies}
\longrightarrow
\text{target logits / drafter outputs / rollout data}.
$$

Thus ReSpec does not need a separate target-data-generation job just to obtain KD targets. The rollout continuously exposes how the **current actor** would score the prefixes and candidates encountered by the drafter.

These observations are stored in a replay buffer and periodically used to train the drafter:

$$
p_{\theta_t}
\overset{\text{online KD}}{\longrightarrow}
q_{\phi_t}.
$$

With ReSpec using a replay buffer, you end up with off-policy data. Hence you still need a *real* drafter training pass: inference-time drafter logits are detached, so when optimizing $\phi$, the stored examples must be run through the current drafter again to obtain gradients. For EAGLE-3, this is also where one can perform its training-time-test unrolling, exposing the drafter to its own recursively generated states rather than only target-produced states.

I wonder why they don't just update the draft at a slower pace, and only use on policy data.Imagine holding the drafter fixed for $K$ actor updates:

$$
\phi_k
\xrightarrow[\text{collect SD data}]{K\text{ RL steps}}
D_k
\xrightarrow{\text{train once}}
\phi_{k+1},
$$

then discard $D_k$. 


### NVIDIA - NemoRL SD

NVIDIA's paper is mainly a **systems-integration study**: put lossless SD inside NeMo RL + vLLM and study rollout gains in synchronous and asynchronous RL.

As opposed to the previous works covered, this initializes the drafter towards the actual training distribution by training on generations from the target based on prompts from the subsequent RL dataset. Thus the drafter is explicitly primed toward the distribution expected during subsequent RL. In the main experiments it is then **frozen for the entire RL run**, with draft length $k=3$.

Despite this (frozen), SD remains effective for very long runs: roughly $1.5-1.8\times$ generation speedup across $600-1000$ RL steps, translating to about $1.35-1.41\times$ end-to-end because generation is only $\sim65-72\%$ of total step time.

This is in tension with ReSpec's frozen-EAGLE result, where acceptance falls substantially after only $\sim100$ actor updates. The useful takeaway is therefore not that frozen drafters inevitably become stale, but that staleness seems to be workload- and initialization-dependent.

**Initialization matters.** Replacing a generic chat-domain initialization with policy-generated, in-domain data improves rollout speedup substantially:

$$
1.51\times \rightarrow 1.77\times,
\qquad
1.19\times \rightarrow 1.53\times.
$$

Acceptance improves as well. They also find that online adaption of the drafter, meaning training it as RL progresses, provides little benefit over the properly initialized drafter, at-least over the measured 1000 training steps:

$$
1.77\times\text{ frozen}\rightarrow1.78\times\text{ online}.
$$

#### async interaction

The async ablation disaggregates generation and training and allows limited policy lag. The first-order effect is pipeline overlap:

$$
T_{\mathrm{async}}\approx\max(T_g,T_t)
$$

instead of $T_g+T_t$.

If generation is slower than training, the trainer eventually waits for rollout data. SD reduces this exposed generation time from $10.4$s to $0.6$s and effective step time from $75.0$s to $60.5$s ($1.24\times$):

$$
T'_{\mathrm{async}}\approx\max(T_g/S_{\mathrm{SD}},T_t).
$$

Hence SD improves end-to-end throughput only while generation contributes to the pipeline bottleneck. Once training is slower, further generation speedup mainly buys resource efficiency rather than wall-clock speedup.

Async also changes the serving regime itself. Allowing more policy lag gives generators more freedom to keep consuming prompts while the trainer works, keeping active batches larger for longer. This pushes the rollout engine toward the high-utilization regime where SD has less spare compute to exploit.

#### deployment-scale simulation

Section 4 uses a proprietary simulator to ask how SD behaves as model scale, GPU count, sharding and policy lag change. It models long-tailed rollout lengths and a fixed global rollout batch, so the important quantity becomes the **local active batch per model instance**.

With fixed global batch:

$$
N_{\mathrm{GPU}}\uparrow
\Rightarrow
B_{\mathrm{local}}\downarrow
\Rightarrow
\text{AR utilization}\downarrow
\Rightarrow
\text{more room for SD verification}.
$$

Thus a huge global RL batch can still be a favorable SD regime if it is spread across enough model instances.

Figure 4 shows two useful trends for the frontier-scale model:

$$
N_{\mathrm{GPU}}\uparrow
\Rightarrow
\text{SD rollout speedup generally}\uparrow,
$$

while

$$
\text{policy lag}\uparrow
\Rightarrow
\text{SD rollout speedup generally}\downarrow.
$$

The two effects fit together: more GPUs shrink the local prompt pool and leave compute underutilized; more policy lag decouples generators from trainers and keeps that pool replenished for longer, improving ordinary decoding utilization and reducing SD's relative advantage.

At extreme scale the relationship becomes non-monotonic because sharding and pipeline concurrency interact. The best simulated operating point reaches roughly $3.5\times$ rollout speedup and $2.5\times$ end-to-end.

Across the three papers, the shared systems fact is: **SD is least attractive when ordinary AR decoding already saturates the GPU.** ReSpec adapts the SD configuration with batch size; EfficientRollout runs AR early and SD in the low-batch tail; NVIDIA mostly uses a fixed short draft, but its scale analysis is governed by the same local-batch / under-utilization effect.

**Caveat:** My biggest gripe with this paper, especially compared with EfficientRollout and ReSpec, is that it does not really account for the effect of batch size on SD efficacy. It acknowledges batching effects, but its main experiments use SD continuously with a fixed draft length and report substantial speedups. Yet the broader literature suggests that SD is often unattractive once generation is already throughput-bound at high local batch sizes, because verification can no longer exploit otherwise-idle compute.

This makes the results difficult to interpret. Given fixed inference resources, I would normally first increase generation batch size until the hardware is close to maximally utilized. If the NVIDIA experiments operate below that point, then some of the reported SD gain may simply come from using speculation to fill otherwise-unused compute, rather than improving on an already throughput-optimized AR baseline.

There may of course be reasons why the rollout workload cannot sustain sufficiently large local batches—in which case SD is exactly a useful mechanism for recovering that lost utilization. But because the paper does not report enough about local batch size or hardware utilization in the main experiments, it is hard to tell which regime they are actually in, and therefore hard to take the reported speedups as representative of a well-utilized frontier RL rollout stack.

### Frontier Models
*this was edited in October 2026*

There's been a few scattered reports of use of spec decode in training of frontier models. The details are generally scarce, from what I can tell nobody has touched on the drifting policy problem, it's moreso been reports/acknowledgements of the usage of spec decode in the rollout engines. I wanted to at least mention which labs have reported employing spec decode. For this I'll focus on those reported using it in async rl frameworks, as the usage in synchronous rl is considerably more straightforward.

**Nemotron 3 Ultra (§3.6.1, pp. 31–32.)**
They run a one-step off-policy async RL with MTP-based spec decode on the rollout engines. The report describes a controlled sweep over zero, three, five, and seven draft tokens, with five tokens resulting in a 1.46x reduction in mean rollout-generation wall time. Most of the benefit comes from the slowest trajectories, which often determine how quickly the (one-step) async training loop advances.

**MiMo-V2.6 (§6)**
Describe a large-batch async RL with rollouts running DFlash spec decode by default. Block size of six. The drafter is initialized from the SFT policy and fine-tuned on early RL rollout logs to track the changing policy distribution.

**MiniMax M2.5 (§2.1, 3.1 and 3.3)**
Use a MTP-based speculative decoding in the rollout engine. Notably, the MTP heads are continuously fine-tuned using a Top-K KL loss to stay aligned with the evolving RL policy.
