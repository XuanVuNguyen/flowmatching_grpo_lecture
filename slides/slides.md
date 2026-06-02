---
title: Flow Matching & GRPO
info: |
  EPFL-themed Slidev template. Lecture notes for the
  Flow Matching + GRPO module of the RL course.
author: Speaker Name
theme: default
class: epfl-cover
mdc: true
transition: slide-left
drawings:
  persist: false
fonts:
  sans: 'Arial'
  mono: 'JetBrains Mono'
layout: cover
preparedBy:
  - Bohdan Naida
  - Vu Nguyen
---

# Flow Matching & GRPO
## E-568 Theory and Methods of Reinforcement Learning taught at EPFL by Prof. Volkan Cevher

<!--
The cover layout takes `event`, `venue`, `date`, `speaker`, and `speakerTitle`
props. Override `image` to swap the background photo:
  layout: cover
  image: /epfl/campus.jpeg
-->

---
layout: outline
---

1. Foundations of Flow Matching
2. Brownian Motion & Stochastic Differential Equations
3. GRPO & Flow Matching
4. Notebook & References

---
layout: section
part: 1
---

# Foundations of Flow Matching

---
class: compact roomy
---

# Flow Matching, General Idea

**Goal**: generate samples from a complex distribution $p_{\text{data}}$ (images, videos, molecules, ...) that we only access through examples

**Idea**: learn to *continuously transport* a simple distribution (e.g. Gaussian noise) into the data distribution over time $t \in [0, 1]$

<div class="note">

$$
\underbrace{p_0 = \mathcal{N}(0, I)}_{\text{easy to sample}} \;\;\xrightarrow{\;\;\text{learned transport}\;\;}\;\; \underbrace{p_1 = p_{\text{data}}}_{\text{what we want}}
$$

</div>

**Why learn a velocity instead of the map itself?** We could try to learn the noise-to-data map directly, but it's globally complex. One function would have to encode the entire transformation. Learning a vector field instead means the network answers only a *local* question: where to move from *this* point at *this* time. The global transformation then emerges by integrating many easy local decisions.

<img :src="'./figures/noise_to_data.gif?v=2'" class="mx-auto block" style="height: 140px; width: auto;" />

To make this precise, we need to talk about how points *move* in space over time. The next few slides define **ODEs, vector fields, flows**, then we return to flow matching itself.

---
class: compact
---

# Ordinary Differential Equations

<div class="definition">

An **ODE** is an equation connecting an unknown function $X : I \to \mathbb{R}^d$ to its derivatives, where $I \subseteq \mathbb{R}$ is an interval. A **first-order ODE** has the form
$$
\frac{dX_t}{dt} = f(X_t,\, t), \qquad f : \mathbb{R}^d \times I \to \mathbb{R}^d.
$$

</div>

<div class="definition">

A **solution**, called a **trajectory**, is a differentiable function $X : I \to \mathbb{R}^d,\; t \mapsto X_t$ that satisfies the ODE pointwise.

</div>

**Why first-order?** In flow matching we specify velocities directly; that is what the neural network outputs, so first-order is the right level of description. We take $I = [0, 1]$ and interpret $t$ as time

<img :src="'./figures/ode_trajectory.gif'" class="mx-auto block" style="height: 130px; width: auto;" />

<div class="example">

A water particle flowing downstream: at each time $t$, $X_t$ gives its position in the river.

</div>

---
class: compact
---

# Vector Field

<div class="definition">

A **vector field** is a function that, for every time $t$ and location $x$, returns a vector $u_t(x) \in \mathbb{R}^d$ specifying a velocity in space:
$$
u : \mathbb{R}^d \times I \to \mathbb{R}^d, \qquad (x, t) \mapsto u_t(x).
$$

</div>

<div class="vf-row">
<div class="vf-left">

Every (sufficiently regular) vector field defines an **ODE**

Paired with an initial condition, it constitutes an **initial value problem (IVP)**:

- $\dot{X}_t = u_t(X_t)$: the trajectory follows the field
- $X_0 = x_0$: the trajectory starts at an initial point

</div>
<div class="vf-right">

<img :src="'./figures/vector_field.gif?v=2'" style="height: 240px; width: auto;" />

</div>
</div>

<div class="theorem">

**Picard–Lindelöf.** If $u_t$ is Lipschitz in $x$ and continuous in $t$, the IVP has a unique solution for every $x_0$. Neural networks satisfy this condition in practice, so we'll assume it without further comment

</div>

---
layout: two-cols
class: compact
---

# Flow

::left::

<div class="definition">

A flow is a one-parameter family of functions $\{\Psi_t\}_{t \in [0, 1]}$, where each $\Psi_t: \mathbb{R}^d \to \mathbb{R}^d$ maps a starting point to its position at time $t$

Equivalently, the family folds into a single combined map:
$$
\Psi: \mathbb{R}^d \times [0, 1] \to \mathbb{R}^d, \qquad (x_0, t) \to \Psi_t(x_0)
$$

The flow is defined by an ODE: for every point $x_0$:
$$
\frac{d}{dt} \Psi_t(x_0) = u_t(\Psi_t(x_0)), \qquad \Psi_0(x_0)=x_0
$$
</div>

So the flow is not a single solution but all of them at once, one trajectory per each starting point. Fixing $x_0$ recovers an initial trajectory, $X_t = \Psi_t(x_0)$ with $X_0 = x_0$

The flow also transports distributions, not just points:

If $X_0 \sim p_0$ then $X_t = \Psi_t(X_0)$ has distribution $p_t$. So, a vector field $u_t$ defines an entire path of distributions $\{p_t\}_{t \in [0, 1]}$ interpolating between $p_0$ and $p_1$

::right::

<img :src="'./figures/flow.gif'" class="mx-auto block" style="width: 100%; height: auto; max-height: 360px;" />

<p style="text-align: center; font-size: 0.8rem; color: #6E6967; margin-top: 0.4em;">
<span style="color: #1f77b4; font-weight: 700;">■</span> flow Ψ<sub>t</sub> (warped grid) &nbsp;&nbsp;
<span style="color: #f4c20d; font-weight: 700;">→</span> velocity field u<sub>t</sub>
</p>

---
class: compact
---

# From Flows to Flow Matching

In **flow matching** we re-use the machinery above to transport noise into data on $I = [0, 1]$:

- A **trajectory** $X_t$ is the path of a single noisy sample as it is transformed into a data sample
- The **vector field** $u_t$ is what we learn: a neural network trained to point toward data
- The **flow** $\Psi_t$ carries *every* noise point to a data point at once; it is the generative map

<img :src="'./figures/noise_to_data_vector.gif'" class="mx-auto block" style="height: 300px; width: auto;" />

---
class: compact
---

# Sampling: Simulating an ODE

In general it is intractable to compute the flow explicitly (unless $u_t$ is very simple, e.g. constant)


To simulate the ODE we use numerical methods. The simplest and one of the most effective is the **Euler method**:

$$
X_{t+h} = X_t + h \cdot u_t(X_t), \qquad t \in \{0, h, 2h, \ldots, (N-1)h\}
$$

where $N \in \mathbb{N}$ is the number of simulation steps and $h = \tfrac{1}{N}$ is the step size.

<img :src="'./figures/euler_sampling.gif'" class="mx-auto block" style="height: 280px; width: auto;" />

**Key property for RL**: randomness comes only from the initial noise $x_0 \sim p_0$, every Euler step is deterministic. This will be a problem when we want to *explore*. We'll fix it in the GRPO setup by injecting stochasticity as each step

---
class: compact roomy
---

# Interpolation Schedules & Training

We need paths connecting noise $x_0 \sim p_0$ to data $x_1 \sim p_{\text{data}}$. A general construction writes $x_t = \alpha_t\, x_0 + \beta_t\, x_1$, with the schedule $(\alpha_t, \beta_t)$ chosen so that $x_0$ is pure noise and $x_1$ is pure data. Two common choices:

<div class="note">

- **Linear**: $\alpha_t = 1-t,\ \beta_t = t$ (straight line from $x_0$ to $x_1$).
- **Variance-preserving**: $\alpha_t = \cos(\tfrac{\pi t}{2}),\ \beta_t = \sin(\tfrac{\pi t}{2})$.

</div>

Each point $x_t$ has a well-defined velocity $u_t(x_t \mid x_0, x_1) = \dot{\alpha}_t\, x_0 + \dot{\beta}_t\, x_1$. For the linear schedule, $\dot{\alpha}_t = -1$ and $\dot{\beta}_t = 1$, so this simplifies to $x_1 - x_0$.

**For the rest of this lecture we use linear interpolation**

To train, sample $x_0 \sim p_0,\ x_1 \sim p_{\text{data}},\ t \sim \mathcal{U}[0, 1]$, form $x_t$, and regress the network's predicted velocity onto the path velocity:

<div class="note">

$$
\mathcal{L}(\theta) = \mathbb{E}\bigl[\,\|u_t^{\theta}(x_t) - (x_1 - x_0)\|^2\,\bigr]
$$

</div>

**Intuition**: Each example sees only one noise–data pair, but in expectation the network learns the *marginal* velocity at $x_t$ — the average direction transporting noise into data. This gives a good generative model, but the target $x_1 - x_0$ knows nothing about reward or preference

<div class="terminology">

In the diffusion literature this is usually called a **noise schedule**, parameterized as $x_t = \alpha_t\, x_1 + \sigma_t\, \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, I)$.

</div>

---
layout: section
part: 2
---

# Brownian motion & Stochastic Differential Equations (SDE)

---
class: compact tight
---

# From ODE to SDE in Practice

We discretise the SDE with the **Euler–Maruyama** scheme. Comparing one step of each:

<div class="note">

$$
\underbrace{X_{t+h} = X_t + h\, u^\theta_t(X_t)}_{\text{ODE (deterministic)}}
\qquad\longrightarrow\qquad
\underbrace{X_{t+h} = X_t + h\, b^\theta_t(X_t) + \sigma\sqrt{h}\,\varepsilon_t}_{\text{SDE (stochastic)}}
$$

</div>

with $\varepsilon_t \sim \mathcal{N}(0, I)$ drawn **independently at every step**.

- The noise scale $\sigma$ controls exploration: $\sigma \to 0$ recovers the ODE; larger $\sigma$ gives noisier, more diverse rollouts.
- The drift $b^\theta_t$ is the **learned velocity field corrected by the score** so that the marginals $p_t$ are preserved:

<div class="note">

$$
b^\theta_t(x) \;=\; u^\theta_t(x) \;+\; \tfrac{\sigma^2}{2}\, \nabla_x \log p_t(x)
$$

</div>

- For **flow** with linear interpolant $X_t = (1-t) X_0 + t X_1$ ($X_0 \sim \mathcal{N}(0, I)$ noise, $X_1 \sim p_{\text{data}}$), the score can be re-expressed purely in terms of $u^\theta_t$, giving the closed-form drift ([see Appendix A](/appendix-rectified-flow)):

<div class="note">

$$
b^\theta_t(x) \;=\; u^\theta_t(x) \;+\; \frac{\sigma_t^2}{2(1-t)}\,\bigl(t\, u^\theta_t(x) - x\bigr)
$$

</div>

- Each Euler step is therefore a **Gaussian transition**:

<div class="note">

$$
\pi_\theta(X_{t+h} \mid X_t) = \mathcal{N}\!\bigl(X_t + h\, b^\theta_t(X_t),\ \sigma^2 h\, I\bigr)
$$

</div>

This is exactly the policy we will optimise with GRPO: a sequence of $N$ Gaussian actions, one per integration step.

---
class: compact
---

# ODE vs SDE: where exploration comes from

<!-- Same drift $u_t$, same starting point, same step size. The only thing that changes across panels is the noise scale $\sigma$ injected at each Euler step. -->

<img :src="'./figures/ode_vs_sde.gif'" class="mx-auto block" style="height: 360px; width: auto;" />

<!-- - **Top-left ($\sigma = 0$, ODE):** one trajectory — the rollout is a deterministic function of $X_0$, so a group of rollouts from the same noise would all be identical.
- **Other panels (SDE):** four trajectories from the *same* $X_0$ diverge because each Euler step adds an independent $\sigma\sqrt{h}\,\varepsilon_t$. Increasing $\sigma$ widens the cloud — this is the exploration knob GRPO needs.

The grey dashed curve in each SDE panel is the underlying ODE trajectory, kept as a visual reference so the noise contribution is easy to read off. -->

---
layout: section
part: 3
---

# GRPO & Flow Matching

---
class: compact tight
---

# GRPO refresher

A **critic-free PPO variant**: the baseline comes from a *group* of rollouts that share the same prompt, not from a learned value function.

**Group sampling.** For prompt $s$, draw $G$ completions from the *ref* policy and score each with a verifier:

<div class="note">

$$
a_i \sim \pi_{\theta_{\text{ref}}}(\cdot \mid s), \qquad r_i = r_{\text{ver}}(s, a_i), \qquad i = 1, \ldots, G.
$$

</div>

The **group-relative advantage** $\hat{A}_i$ rescales each reward by the group mean and std (formula on the next-to-last slide).

**Token ratio.** For each token $a_{i,t}$ of completion $a_i$,

<div class="note">

$$
\rho_{i,t}(\theta) \;:=\; \frac{\pi_\theta(a_{i,t} \mid s,\, a_{i,<t})}{\pi_{\theta_{\text{ref}}}(a_{i,t} \mid s,\, a_{i,<t})}.
$$

</div>

**Full GRPO objective** — PPO-clipped surrogate plus a KL anchor to a frozen reference $\pi_{\text{ref}}$:

<div class="note">

$$
\max_\theta \;\; \mathbb{E}_{s,\,a_{1:G} \sim \pi_{\theta_{\text{ref}}}}\!\Biggl[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T_i}\sum_{t=1}^{T_i} \min\!\bigl(\rho_{i,t}(\theta)\hat{A}_i,\; \operatorname{clip}\!\bigl(\rho_{i,t}(\theta),\, 1-\epsilon_{\text{clip}},\, 1+\epsilon_{\text{clip}}\bigr)\hat{A}_i\bigr)\Biggr] \;-\; \beta\,\hat{D}_{\text{KL}}\!\bigl(\pi_\theta \,\Vert\, \pi_{\text{ref}}\bigr)
$$

</div>

- Above-baseline siblings get $\hat{A}_i > 0$ and are pushed up; below-baseline ones get $\hat{A}_i < 0$ and are suppressed.
- Clipping at radius $\epsilon_{\text{clip}}$ stops the per-token ratio from drifting too far in one update — the standard PPO safeguard.
- The KL penalty anchors to $\pi_{\text{ref}}$ (a fixed snapshot), **not** to $\pi_{\theta_{\text{ref}}}$ — the same pattern as RLHF.

---
class: compact tight
---

# GRPO with Gaussian Policies $\Rightarrow$ Closed-Form KL

Policy at each step is Gaussian with covariance $\sigma_t^2 h\, I$ **that does not depend on $\theta$**. Only the *mean* moves with $\theta$

With $\mu_t^\theta = X_t + h\, b^\theta_t(X_t)$ and $\mu_t^{\text{ref}} = X_t + h\, b^{\text{ref}}_t(X_t)$

The per-step log-ratio reduces to a difference of squared distances:

<div class="note">

$$
\log\frac{\pi_\theta(X_{t+h}\mid X_t)}{\pi_{\text{ref}}(X_{t+h}\mid X_t)}
\;=\; \frac{\lVert X_{t+h} - \mu_{\text{ref}} \rVert^2 \;-\; \lVert X_{t+h} - \mu_\theta \rVert^2}{2\sigma_t^2 h}
$$

</div>

Taking the expectation $\mathrm{KL} = \mathbb{E}_{X \sim \pi^\theta}[\log r_t(X)]$ results in the **closed form** ([see Appendix B](/appendix-closed-form-kl)):

<div class="note">

$$
\mathrm{KL}\!\bigl(\pi^\theta(\cdot\mid X_t)\,\Vert\,\pi^{\text{ref}}(\cdot\mid X_t)\bigr)
\;=\; \frac{\lVert \mu_t^\theta - \mu_t^{\text{ref}} \rVert^2}{2\sigma_t^2 h}.
$$

</div>

Substituting $\mu_t = X_t + h\, b_t$ and the flow drift $b_t = u_t + \tfrac{\sigma_t^2}{2(1-t)}(t\, u_t - x)$, the $-x$ score-correction term **cancels in the difference**:

<div class="note">

$$
\mu_t^\theta - \mu_t^{\text{ref}}
\;=\; h\,\bigl(b^\theta_t(X_t) - b^{\text{ref}}_t(X_t)\bigr)
\;=\; h\, C_t\,\bigl(u^\theta_t(X_t) - u^{\text{ref}}_t(X_t)\bigr),
\qquad
C_t \;\triangleq\; 1 + \frac{t\,\sigma_t^2}{2(1-t)}.
$$

</div>

So the per-step KL reduces to a **time-dependent** weight times the squared residual of the learned velocity:

<div class="note">

$$
\mathrm{KL}\!\bigl(\pi^\theta(\cdot \mid X_t)\,\Vert\,\pi^{\text{ref}}(\cdot \mid X_t)\bigr)
\;=\; \frac{h\, C_t^2}{2\sigma_t^2}\,\bigl\lVert u^\theta_t(X_t) - u^{\text{ref}}_t(X_t) \bigr\rVert^2.
$$

</div>

--- 
class: compact tight middle
---

# GRPO with Gaussian Policies $\Rightarrow$ Closed-Form KL (cont.)

**Group-relative advantage** (recall, $G$ rollouts scored by the verifier):

<div class="note">

$$
\hat{A}_i \;=\; \frac{r_i - \bar{r}_s}{\hat{\sigma}_s + \epsilon_{\text{std}}},
\qquad
\bar{r}_s = \tfrac{1}{G}\!\sum_{i=1}^{G} r_i,
\qquad
\hat{\sigma}_s^2 = \tfrac{1}{G}\!\sum_{i=1}^{G}(r_i - \bar{r}_s)^2.
$$

</div>

**Time-dependent KL weight** (recall):

<div class="note">

$$
C_t \;\triangleq\; 1 + \frac{t\,\sigma_t^2}{2(1-t)}.
$$

</div>

Aggregating the per-step terms across all steps of the sampled trajectory, the objective becomes:

<div class="note">

$$
\mathcal{L}_{\text{GRPO}}
= \mathbb{E}\Bigl[\sum_{k}\min\bigl(r_{t_k} \hat{A},\, \operatorname{clip}(r_{t_k}, 1{-}\epsilon, 1{+}\epsilon)\hat{A}\bigr)\Bigr]
\;-\;\beta \sum_{k}\frac{h\, C_{t_k}^2}{2\sigma_{t_k}^2}\bigl\lVert u^\theta_{t_k}(X_{t_k}) - u^{\text{ref}}_{t_k}(X_{t_k})\bigr\rVert^2
$$

</div>

<!-- > Closed-form KL is the practical benefit of staying Gaussian: low variance, no Monte-Carlo simulation, and gradients flow directly into the velocity network $u^\theta_t$. -->

---
class: compact
---
# Notebook to Practice

<div class="practice-steps">

<div class="practice-step">
<div class="step-num">1</div>
<div class="step-body">
<div class="step-title">Build the pipeline, end to end</div>
Train flow matching on a 2D checkerboard distribution, swap the ODE for an SDE, and let GRPO steer it with a predefined reward
</div>
</div>

<div class="practice-step">
<div class="step-num">2</div>
<div class="step-body">
<div class="step-title">Design your custom reward</div>
Smooth, sparse, multi-modal, repulsive. See how reward shapes the policy and find out what the policy can (and can't) learn
</div>
</div>

<div class="practice-step">
<div class="step-num">3</div>
<div class="step-body">
<div class="step-title">Sweep across KL coefficients</div>
Watch the trade-off between maximizing the reward and staying close to the prior
</div>
</div>

</div>

---
class: compact
---

# References

- Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y. K., Wu, Y., Guo, D. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv:2402.03300. [arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)

- Liu, J., Liu, G., Liang, J., Li, Y., Liu, J., Wang, X., Wan, P., Zhang, D., Ouyang, W. (2025). *Flow-GRPO: Training Flow Matching Models via Online RL*. arXiv:2505.05470. [arxiv.org/abs/2505.05470](https://arxiv.org/abs/2505.05470)

- Holderrieth, P., Erives, E. (2026). *Introduction to Flow Matching and Diffusion Models*. arXiv:2506.02070. [diffusion.csail.mit.edu](https://diffusion.csail.mit.edu/)

---
layout: section
part: A
---

# Appendix

---
class: compact tight
routeAlias: appendix-rectified-flow
---

# Appendix A: The Flow Drift $b^\theta_t$

**Recall.** $b^\theta_t(x) = u^\theta_t(x) + \tfrac{\sigma_t^2}{2}\, \nabla_x \log p_t(x)$.

**Setup.** Gaussian mixture path $X_t = \alpha_t X_0 + \beta_t X_1$ with $X_0 \sim \mathcal{N}(0, I)$, $X_1 \sim p_{\text{data}}$, and boundary $(\alpha_0,\beta_0) = (1, 0)$, $(\alpha_1,\beta_1) = (0, 1)$. Conditioning on the data endpoint gives a tractable Gaussian:

<div class="note">

$$
p_t(x \mid x_1) \;=\; \mathcal{N}\!\bigl(\beta_t\, x_1,\ \alpha_t^2 I\bigr).
$$

</div>

**Step 1.** Differentiating the interpolant and conditioning on $X_t = x$:

<div class="note">

$$
u_t(x) \;=\; \dot\alpha_t\, \mathbb{E}[X_0 \mid X_t = x] \;+\; \dot\beta_t\, \mathbb{E}[X_1 \mid X_t = x].
$$

</div>

Using the consistency constraint $\alpha_t\mathbb{E}[X_0|X_t] + \beta_t \mathbb{E}[X_1|X_t] = x$ to eliminate $\mathbb{E}[X_0|X_t]$:

<div class="note">

$$
u_t(x) \;=\; \tfrac{\dot\alpha_t}{\alpha_t}\, x \;+\; \tfrac{\alpha_t \dot\beta_t - \beta_t \dot\alpha_t}{\alpha_t}\, \mathbb{E}[X_1 \mid X_t = x]. \qquad(\star)
$$

</div>

**Step 2.** Differentiate the conditional Gaussian: $\nabla_x \log p_t(x|x_1) = -(x - \beta_t x_1)/\alpha_t^2$. Marginalising via $\nabla \log p_t(x) = \mathbb{E}_{X_1 \mid X_t = x}[\nabla \log p_t(x | X_1)]$ (differentiation under the integral + Bayes):

<div class="note">

$$
\nabla_x \log p_t(x) \;=\; \frac{\beta_t\, \mathbb{E}[X_1 \mid X_t = x] \;-\; x}{\alpha_t^2}. \qquad(\star\star)
$$

</div>

Steps 1 and 2 give us $u_t$ and $\nabla_x \log p_t$ each as an affine function of the *same* unknown $\mathbb{E}[X_1 \mid X_t = x]$. Continued on the next slide.

---
class: compact tight
---

# Appendix A (cont.): The Flow Drift $b^\theta_t$$

**Step 3.** Solving $(\star\star)$ for $\mathbb{E}[X_1|X_t]$ and substituting into $(\star)$ gives a relation for *any* Gaussian mixture path:

<div class="note">

$$
\boxed{\;\nabla_x \log p_t(x) \;=\; \frac{\beta_t\, u_t(x) \;-\; \dot\beta_t\, x}{\alpha_t\,(\alpha_t \dot\beta_t - \beta_t \dot\alpha_t)}\;}.
$$

</div>

> This relationship between score function and vector field is true for any choice of the schedulers $\alpha_t$ and $\beta_t$.

**Step 4.** Plug in $\alpha_t = 1-t,\ \beta_t = t$ (so $\dot\alpha_t = -1,\ \dot\beta_t = 1,\ \alpha_t\dot\beta_t - \beta_t\dot\alpha_t = 1$):

<div class="note">

$$
\nabla_x \log p_t(x) \;=\; \frac{t\, u^\theta_t(x) - x}{1 - t}
\quad\Longrightarrow\quad
b^\theta_t(x) \;=\; u^\theta_t(x) \;+\; \frac{\sigma_t^2}{2(1-t)}\bigl(t\, u^\theta_t(x) - x\bigr).
$$

</div>

---
class: compact tight
routeAlias: appendix-closed-form-kl
---

# Appendix B: The closed-form KL

**Setup.** Two isotropic Gaussian per-step policies with **identical** covariance:

<div class="note">

$$
\pi_\theta(\cdot \mid X_t) = \mathcal{N}(\mu_t^\theta,\ \Sigma),
\qquad
\pi_{\text{ref}}(\cdot \mid X_t) = \mathcal{N}(\mu_t^{\text{ref}},\ \Sigma),
\qquad
\Sigma = \sigma_t^2 h\, I.
$$

</div>

**Step 1. log-ratio.** The Gaussian log-density is $\log \mathcal{N}(x; \mu, \Sigma) = -\tfrac{1}{2}(x-\mu)^\top \Sigma^{-1}(x-\mu) + \text{const}(\Sigma)$. The $\Sigma$-dependent normalizers are *the same* under $\theta$ and ref, so they cancel:

<div class="note">

$$
\log \pi_\theta(X) - \log \pi_{\text{ref}}(X)
\;=\; \tfrac{1}{2\sigma_t^2 h}\bigl[\lVert X - \mu_t^{\text{ref}} \rVert^2 - \lVert X - \mu_t^\theta \rVert^2\bigr]
$$

</div>

**Step 2. KL by definition.** $\mathrm{KL}(\pi_\theta \,\Vert\, \pi_{\text{ref}}) = \mathbb{E}_{X \sim \pi_\theta}\bigl[\log \pi_\theta(X) - \log \pi_{\text{ref}}(X)\bigr]$. Parametrize $\pi_\theta$ as $X = \mu_t^\theta + \sigma_t \sqrt{h}\, \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, I)$:

<div class="note">

$$
X - \mu_t^\theta = \sigma_t \sqrt{h}\, \varepsilon,
\qquad
X - \mu_t^{\text{ref}} = (\mu_t^\theta - \mu_t^{\text{ref}}) + \sigma_t \sqrt{h}\, \varepsilon
$$

</div>

**Step 3. expand the squared norms.** Using $\lVert a + b \rVert^2 = \lVert a \rVert^2 + 2 a^\top b + \lVert b \rVert^2$ (the $\sigma_t^2 h\,\lVert\varepsilon\rVert^2$ terms cancel):

<div class="note">

$$
\lVert X - \mu_t^{\text{ref}} \rVert^2 - \lVert X - \mu_t^\theta \rVert^2
\;=\; \lVert \mu_t^\theta - \mu_t^{\text{ref}} \rVert^2 \;+\; 2 \sigma_t \sqrt{h}\,(\mu_t^\theta - \mu_t^{\text{ref}})^\top \varepsilon
$$

</div>

**Step 4. take the expectation.** $\mathbb{E}[\varepsilon] = 0$, so the cross-term vanishes:

<div class="note">

$$
\mathbb{E}_{X \sim \pi_\theta}\bigl[\lVert X - \mu_t^{\text{ref}} \rVert^2 - \lVert X - \mu_t^\theta \rVert^2\bigr]
\;=\; \lVert \mu_t^\theta - \mu_t^{\text{ref}} \rVert^2
$$

</div>

Dividing by $2\sigma_t^2 h$ recovers the formula on the main slide:

<div class="note">

$$
\boxed{\;\mathrm{KL}\!\bigl(\pi_\theta \,\Vert\, \pi_{\text{ref}}\bigr) \;=\; \frac{\lVert \mu_t^\theta - \mu_t^{\text{ref}} \rVert^2}{2\sigma_t^2 h}\;}
$$

</div>