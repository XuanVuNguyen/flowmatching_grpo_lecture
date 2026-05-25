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
class: compact
---

# Ordinary Differential Equations (ODEs)

An **ODE** is an equation relating an unknown function $X : I \to \mathbb{R}^d$ to its derivatives, where $I \subseteq \mathbb{R}$ is an interval

A **first-order ODE** has the form:
$$
\frac{dX_t}{dt} = f(X_t,\, t), \qquad f : \mathbb{R}^d \times I \to \mathbb{R}^d
$$

A **solution** is a differentiable function, called a **trajectory**, that maps each $t \in I$ to a location $X_t \in \mathbb{R}^d$ and satisfies the ODE:
$$
X : I \to \mathbb{R}^d, \qquad t \mapsto X_t
$$

In flow matching, we take $I = [0, 1]$ and interpret $t$ as time.

<img :src="'/figures/ode_trajectory.gif'" class="mx-auto block" style="width: 70%; height: auto;" />

**Example**: A water particle flowing downstream, at each time $t$, $X_t$ gives its position in the river

---
class: compact
---

# Vector Field

A **vector field** is a function that, for every time $t$ and location $x$, returns a vector $u_t(x) \in \mathbb{R}^d$ specifying a velocity in space:
$$
u : \mathbb{R}^d \times I \to \mathbb{R}^d, \qquad (x, t) \mapsto u_t(x)
$$

Every vector field defines an **ODE**. Paired with an initial condition, it constitutes an **initial value problem (IVP)**:

- The trajectory **follows the vector field**:
$$\qquad \frac{d}{dt} X_t = u_t(X_t)$$
- The trajectory **starts at an initial point**$:
$$X_0 = x_0$$

The vector field generates the dynamics, and the trajectory is what you get by following it

<img :src="'/figures/vector_field.gif'" class="mx-auto block" style="width: 70%; height: auto;" />

---
class: compact
---

# Flow

A **flow** is a function that, for every initial point $x_0$ and time $t$, returns the position at time $t$ of the trajectory starting from $x_0$:
$$
\Psi : \mathbb{R}^d \times [0, 1] \to \mathbb{R}^d, \qquad (x_0, t) \mapsto \Psi_t(x_0)
$$

The flow is itself a solution of the ODE, one for every starting point:
$$
\frac{d}{dt} \Psi_t(x_0) = u_t(\Psi_t(x_0)), \qquad \Psi_0(x_0) = x_0
$$

While a trajectory is a single curve, the flow **tracks all trajectories at once**. For any initial condition $X_0 = x_0$, the corresponding trajectory is recovered by fixing $x_0$: $X_t = \Psi_t(x_0)$

<img :src="'/figures/flow.gif'" class="mx-auto block" style="height: 280px; width: auto;" />

---
layout: two-cols
---

# Conditional vs Marginal Paths

The **conditional path** $p_t(x \mid x_1)$ interpolates between $p_0$ and a
single data point $x_1$. The **marginal path** $p_t(x)$ is the data-averaged
version:

$$
p_t(x) = \int p_t(x \mid x_1)\, p_{\text{data}}(x_1)\, dx_1
$$

::right::

# Conditional vs Marginal Fields

The corresponding **conditional vector field** $u_t(x \mid x_1)$ pushes mass
along the conditional path. The marginal field is its data-weighted average:

$$
u_t(x) = \mathbb{E}_{x_1 \sim p_t(\cdot \mid x)}\bigl[u_t(x \mid x_1)\bigr]
$$

> Training $v_\theta \approx u_t$ on the conditional regression target gives
> the same gradient as regressing on the (intractable) marginal target.

---
layout: section
chapter: 2
---

# From GRPO to Flow Matching

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
- Advantage `Â` computed from a **group** of rollouts — no critic needed
- KL-to-reference is added as a penalty (or removed — see ablation later)

---
class: compact
---

# Why ODE is the wrong substrate for GRPO

```ts {monaco}
// Sampling an ODE is deterministic given x(0).
// That kills exploration — every rollout from the same noise is identical.
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

Use the `full-image` layout for visually dominant slides — diagrams, hero
images, single equations.

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
layout: end
event: EPFL RL Course
speaker: Speaker Name
speakerTitle: Lecturer
---

# Thank you!

Questions, comments, or homework debugging — happy to take them now.
