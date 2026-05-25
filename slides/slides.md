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

# Ordinary Differential Equations

A trajectory $x(t)$ is the solution of an ODE driven by a **vector field** $v_\theta$:

$$
\frac{d x(t)}{dt} = v_\theta\bigl(x(t),\, t\bigr), \qquad x(0) \sim p_0
$$

- $p_0$ is a tractable noise distribution
- $p_1$ is the data distribution we want to sample from
- $v_\theta$ is parameterised by a neural network

This is the **default content layout** — header bar, EPFL logo, title/speaker
in the footer and an auto page counter.

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
