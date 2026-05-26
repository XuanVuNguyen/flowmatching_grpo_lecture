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

1. ODEs and trajectories
2. Probability paths & vector fields
3. GRPO refresher
4. From GRPO to flow matching
5. Brownian motion & SDEs
6. Putting the algorithm together
7. Ablations: KL vs no-KL
8. Future work & homework

---
layout: section
chapter: 1
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

<img :src="'/figures/noise_to_data.gif?v=2'" class="mx-auto block" style="height: 140px; width: auto;" />

To make this precise, we need to talk about how points *move* in space over time. The next few slides define **ODEs, vector fields, flows**, then we return to flow matching itself.

---
class: compact
---

# Ordinary Differential Equations

<div class="definition">

An **ODE** is an equation relating an unknown function $X : I \to \mathbb{R}^d$ to its derivatives, where $I \subseteq \mathbb{R}$ is an interval. A **first-order ODE** has the form
$$
\frac{dX_t}{dt} = f(X_t,\, t), \qquad f : \mathbb{R}^d \times I \to \mathbb{R}^d.
$$

</div>

<div class="definition">

A **solution**, called a **trajectory**, is a differentiable function $X : I \to \mathbb{R}^d,\; t \mapsto X_t$ that satisfies the ODE pointwise.

</div>

**Why first-order?** In flow matching we specify velocities directly, that is what the neural network outputs, so first-order is the right level of description. We take $I = [0, 1]$ and interpret $t$ as time

<img :src="'/figures/ode_trajectory.gif'" class="mx-auto block" style="height: 130px; width: auto;" />

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

Every (sufficiently regular) vector field defines an **ODE**. Paired with an initial condition, it constitutes an **initial value problem (IVP)**:

- $\dot{X}_t = u_t(X_t)$: the trajectory follows the field
- $X_0 = x_0$: the trajectory starts at an initial point

</div>
<div class="vf-right">

<img :src="'/figures/vector_field.gif?v=2'" style="height: 240px; width: auto;" />

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

The **flow** $\Psi_t(x_0)$ is the position, at time $t$, of the trajectory starting from $x_0$:
$$
\Psi : \mathbb{R}^d \times [0, 1] \to \mathbb{R}^d.
$$

It is itself a solution of the ODE, one curve for every starting point:
$$
\tfrac{d}{dt} \Psi_t(x_0) = u_t(\Psi_t(x_0)), \quad \Psi_0(x_0) = x_0.
$$

</div>

For any initial condition $X_0 = x_0$, the corresponding trajectory is recovered by fixing $x_0$: $X_t = \Psi_t(x_0)$. While a trajectory is a single curve, the flow tracks all of them at once

The flow transports distributions, not just points: if $X_0 \sim p_0$ then $X_t = \Psi_t(X_0)$ has distribution $p_t$. So, a vector field $u_t$ defines an entire path $\{p_t\}_{t \in [0, 1]}$ interpolating between $p_0$ and $p_1$

::right::

<img :src="'/figures/flow.gif'" class="mx-auto block" style="width: 100%; height: auto; max-height: 360px;" />

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

<img :src="'/figures/noise_to_data_vector.gif'" class="mx-auto block" style="height: 300px; width: auto;" />

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

<img :src="'/figures/euler_sampling.gif'" class="mx-auto block" style="height: 280px; width: auto;" />

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

Each point $x_t$ has a well-defined velocity $u_t(x_t \mid x_0, x_1) = \dot{\alpha}_t\, x_0 + \dot{\beta}_t\, x_1$. For the linear schedule, $\dot{\alpha}_t = -1$ and $\dot{\beta}_t = 1$, so this simplifies to $x_1 - x_0$ — the displacement vector from noise to data.

**For the rest of this lecture we use linear interpolation unless stated otherwise.**

To train, sample $x_0 \sim p_0,\ x_1 \sim p_{\text{data}},\ t \sim \mathcal{U}[0, 1]$, form $x_t$, and regress the network's predicted velocity onto the path velocity:

<div class="note">

$$
\mathcal{L}(\theta) = \mathbb{E}\bigl[\,\|u_t^{\theta}(x_t) - (x_1 - x_0)\|^2\,\bigr]
$$

</div>

**Intuition**: Each example sees only one noise–data pair, but in expectation the network learns the *marginal* velocity at $x_t$ — the average direction transporting noise into data. This gives a good generative model, but the target $x_1 - x_0$ knows nothing about reward or preference. **That's where GRPO comes in.**

<div class="terminology">

In the diffusion literature this is usually called a **noise schedule**, parameterized as $x_t = \alpha_t\, x_1 + \sigma_t\, \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, I)$.

</div>

---
class: compact
---

# GRPO refresher

Group Relative Policy Optimisation maximises an advantage-weighted ratio with
a clipping safeguard:

$$
\mathcal{L}_{\text{GRPO}} = \mathbb{E}\Bigl[\min\bigl(r_t \hat{A}_t,\
\operatorname{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\bigr)\Bigr]
$$

- `r_t = π_θ(a|s) / π_ref(a|s)`
- Advantage `Â` computed from a **group** of rollouts, no critic needed
- KL-to-reference is added as a penalty (or removed; see ablation later)

---
class: compact
---

# Why ODE is the wrong substrate for GRPO

```ts {monaco}
// Sampling an ODE is deterministic given x(0).
// That kills exploration: every rollout from the same noise is identical.
const x0 = sampleNoise()
const x1 = odeSolver(velocityField, x0)
```

We need **stochastic** trajectories so the group of rollouts has variance to
score against. Enter SDEs.

---
layout: full-image
image: /epfl/campus.jpeg
---

# Brownian motion & SDEs

$$
dx_t = b(x_t, t)\, dt + \sigma(x_t, t)\, dW_t
$$

Use the `full-image` layout for visually dominant slides: diagrams, hero
images, single equations.

---
class: compact
---

# From ODE to SDE in Practice

We discretise the SDE with the **Euler–Maruyama** scheme. Comparing one step of each:

$$
\underbrace{X_{t+h} = X_t + h\, u^\theta_t(X_t)}_{\text{ODE (deterministic)}}
\qquad\longrightarrow\qquad
\underbrace{X_{t+h} = X_t + h\, b^\theta_t(X_t) + \sigma\sqrt{h}\,\varepsilon_t}_{\text{SDE (stochastic)}}
$$

with $\varepsilon_t \sim \mathcal{N}(0, I)$ drawn **independently at every step**.

- The noise scale $\sigma$ controls exploration: $\sigma \to 0$ recovers the ODE; larger $\sigma$ gives noisier, more diverse rollouts.
- The drift $b^\theta_t$ is the **learned velocity field corrected by the score** so that the marginals $p_t$ are preserved:
$$
b^\theta_t(x) \;=\; u^\theta_t(x) \;+\; \tfrac{\sigma^2}{2}\, \nabla_x \log p_t(x)
$$
- For **rectified flow** (linear interpolant $X_t = (1-t) X_0 + t X_1$), the score can be re-expressed purely in terms of the learned $u^\theta_t$, giving the closed-form drift
$$
b^\theta_t(x) \;=\; -\frac{1}{t}\,\bigl(x + (1-t)\, u^\theta_t(x)\bigr)
\qquad \text{(see appendix)}
$$
- Each Euler step is therefore a **Gaussian transition**:
$$
\pi_\theta(X_{t+h} \mid X_t) = \mathcal{N}\!\bigl(X_t + h\, b^\theta_t(X_t),\ \sigma^2 h\, I\bigr)
$$

This is exactly the policy we will optimise with GRPO: a sequence of $N$ Gaussian actions, one per integration step.

---
class: compact
---

# ODE vs SDE: where exploration comes from

Same drift $u_t$, same starting point, same step size. The only thing that changes across panels is the noise scale $\sigma$ injected at each Euler step.

<img :src="'/figures/ode_vs_sde.gif'" class="mx-auto block" style="height: 360px; width: auto;" />

- **Top-left ($\sigma = 0$, ODE):** one trajectory — the rollout is a deterministic function of $X_0$, so a group of rollouts from the same noise would all be identical.
- **Other panels (SDE):** four trajectories from the *same* $X_0$ diverge because each Euler step adds an independent $\sigma\sqrt{h}\,\varepsilon_t$. Increasing $\sigma$ widens the cloud — this is the exploration knob GRPO needs.

The grey dashed curve in each SDE panel is the underlying ODE trajectory, kept as a visual reference so the noise contribution is easy to read off.

---
class: compact
---

# GRPO with Gaussian Policies $\Rightarrow$ Closed-Form KL

Because every step's policy is Gaussian with **fixed** covariance $\sigma^2 h\, I$, only the *mean* depends on $\theta$. Write $\mu_\theta = X_t + h\, b^\theta_t(X_t)$ and $\mu_{\text{old}} = X_t + h\, b^{\text{old}}_t(X_t)$.

The per-step log-ratio collapses to a difference of squared distances:

$$
\log r_t \;=\; \log\frac{\pi_\theta(X_{t+h}\mid X_t)}{\pi_{\text{old}}(X_{t+h}\mid X_t)}
\;=\; \frac{\lVert X_{t+h} - \mu_{\text{old}} \rVert^2 \;-\; \lVert X_{t+h} - \mu_\theta \rVert^2}{2\sigma^2 h}
$$

and the KL penalty has a **closed form**:

$$
\mathrm{KL}\!\bigl(\pi_\theta(\cdot\mid X_t)\,\Vert\,\pi_{\text{old}}(\cdot\mid X_t)\bigr)
\;=\; \frac{\lVert \mu_\theta - \mu_{\text{old}} \rVert^2}{2\sigma^2 h}
$$

The full GRPO objective summed over the $N$ Euler steps becomes:

$$
\mathcal{L}_{\text{GRPO}}
= \mathbb{E}\Bigl[\sum_{t}\min\bigl(r_t \hat{A},\, \operatorname{clip}(r_t, 1{-}\epsilon, 1{+}\epsilon)\hat{A}\bigr)\Bigr]
\;-\;\beta \sum_{t}\frac{\lVert \mu_\theta - \mu_{\text{old}} \rVert^2}{2\sigma^2 h}
$$

> Closed-form KL is the practical benefit of staying Gaussian: low variance, no Monte-Carlo simulation.

---
layout: section
chapter: 3
---

# Implementation & results

---
class: compact
---

# Reference

| Layout | When to use |
| --- | --- |
| `cover` | Title page |
| `outline` | Table of contents |
| `section` | Red divider between chapters |
| `default` | Standard content |
| `two-cols` | Side-by-side comparison (use `::right::`) |
| `full-image` | Hero image / single equation |
| `end` | Closing thank-you slide |

Reusable bits in `styles/global.css`:

```html
<div class="epfl-cols">
  <div class="epfl-card"><div class="card-title">Left</div>…</div>
  <div class="epfl-card"><div class="card-title">Right</div>…</div>
</div>
```

---
layout: section
chapter: A
---

# Appendix

---
class: compact
---

# Appendix A — Rectified-flow drift $b^\theta_t$

**Setup.** Linear interpolant with $X_0 \sim p_{\text{data}}$ at $t=0$ and $X_1 \sim \mathcal{N}(0, I)$ at $t=1$:
$$
X_t = (1-t)\, X_0 + t\, X_1,
\qquad
p_t(x \mid x_0) = \mathcal{N}\!\bigl((1-t)\, x_0,\ t^2 I\bigr)
$$

**Step 1 — denoiser identity.** The learned velocity satisfies $u^\theta_t(x) = \mathbb{E}[X_1 - X_0 \mid X_t = x]$. Combining with $X_t = (1-t) X_0 + t X_1$ and taking conditional expectations,
$$
\mathbb{E}[X_0 \mid X_t = x] \;=\; x - t\, u^\theta_t(x)
$$

**Step 2 — score via Tweedie.** For the Gaussian mixture $p_t(x) = \int \mathcal{N}(x;\, (1-t) x_0, t^2 I)\, p_{\text{data}}(x_0)\, dx_0$, Tweedie's identity gives
$$
\nabla_x \log p_t(x) \;=\; \frac{(1-t)\, \mathbb{E}[X_0 \mid X_t = x] - x}{t^2}
$$
Substituting Step 1 and simplifying the numerator $(1-t)(x - t\, u^\theta_t) - x = -t\,x - t(1-t)\, u^\theta_t$:
$$
\boxed{\;\nabla_x \log p_t(x) \;=\; -\frac{1}{t}\bigl(x + (1-t)\, u^\theta_t(x)\bigr)\;}
$$

**Step 3 — plug into $b^\theta_t$.** Starting from $b^\theta_t(x) = u^\theta_t(x) + \tfrac{\sigma^2}{2} \nabla_x \log p_t(x)$ and choosing the noise schedule $\sigma^2 = 2t / (1-t)$ used in the paper, the $u^\theta_t$ terms cancel and we obtain
$$
b^\theta_t(x) \;=\; -\frac{1}{t}\bigl(x + (1-t)\, u^\theta_t(x)\bigr).
$$

> The point of Step 2 is that the score is *exactly* re-expressible in the learned velocity — no separate score network is needed.

---
layout: end
event: EPFL RL Course
speaker: Speaker Name
speakerTitle: Lecturer
---

# Thank you!

Questions, comments, or homework debugging. Happy to take them now.
