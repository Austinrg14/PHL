# Projected Hessian Learning (PHL): Training & Evaluation Code

Code and analysis notebooks for the paper **“Projected Hessian Learning (PHL): Fast Curvature Supervision for Accurate Machine-Learning Interatomic Potentials”**.

This repository contains the training and evaluation workflow used to:
- train **ANI-style neural network potentials** with **energy + forces** (E–F),
- add **second-order supervision** via **Hessian–vector products (HVPs)** using **Projected Hessian Learning (PHL)**,
- compare **Gaussian (Hutchinson-type) probing** vs **one-hot (column) probing**,
- benchmark accuracy on **Test / IRC / NMS** datasets and report runtime/speed comparisons.

> You will add: (i) the main notebook(s), and (ii) dataset access instructions/links.

---

## Contents

- [What is PHL?](#what-is-phl)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Dataset access](#dataset-access)
- [Quickstart (Notebook)](#quickstart-notebook)
- [Reproducing paper results](#reproducing-paper-results)
- [Method options](#method-options)
- [Outputs](#outputs)
- [Citing](#citing)
- [License](#license)
- [Contact](#contact)

---

## What is PHL?

**Projected Hessian Learning (PHL)** supervises *projected curvature* of the potential energy surface by training on **HVP targets** instead of full Hessian matrices. For a probe vector \(v\), PHL matches:
\[
\tilde{H}v \approx Hv
\]
where \(H=\nabla^2_{\mathbf{R}} E\) is the reference Hessian and \(\tilde{H}\) is the model Hessian. Probes can be:
- **Gaussian / Hutchinson-type:** \(v \sim \mathcal{N}(0,I)\) (random projections)
- **One-hot / column:** \(v = \sqrt{3N} e_c\) (single column per probe)

PHL provides most of the benefit of curvature supervision at **force-like cost**, avoiding explicit Hessian construction.

---

## Repository layout

Recommended structure (adjust as needed):
