# Projected Hessian Learning (PHL)

Reference implementation of **Projected Hessian Learning (PHL)** — a fast, stochastic curvature-supervision method for training machine-learning interatomic potentials (MLIPs) using **Hessian–vector products (HVPs)** instead of full Hessian matrices.

PHL replaces explicit Hessian supervision with projected curvature targets of the form \(Hv\), enabling second-order training at **force-like** computational cost.

> **Paper:** *Projected Hessian Learning (PHL): Fast Curvature Supervision for Accurate Machine-Learning Interatomic Potentials* (arXiv / submitted to IOP **Machine Learning: Science and Technology (MLST)**)

---

## What’s in this repository

- Training and evaluation code for ANI-style models with:
  - **E–F** (energy + forces) baseline
  - **E–F–HVP (PHL)** using **Gaussian (Hutchinson-type) probing**
  - (optional) **E–F–HVP (one-hot/column)** for comparison
  - (optional) **E–F–H** full Hessian supervision (when full Hessians are available)
- Notebooks/scripts to reproduce:
  - RMSE results on **Test / IRC / NMS**
  - timing comparisons (epoch time / relative speedups)
  - figure generation (main text + SI)

> You will add the main notebook(s) and dataset access instructions (see below).

---

## Getting Started

### Prerequisites

Ensure the following Python packages are installed:

- `torch`
- `torchani`
- `wandb`
- `numpy`
- `tqdm`

> Note: this installation section is intentionally kept the same as in the companion EFH/ANI training README we used as a baseline for these workflows. fileciteturn0file0

### Cloning the Repository

```bash
git clone https://github.com/Austinrg14/PHL.git
cd PHL
```

---

## Dataset access (OpenREACT)

This work uses the **OpenREACT-CHON-FH** datasets (RTP / IRC / NMS with energies, forces, and Hessians).  
Download from Figshare:

```text
https://doi.org/10.6084/m9.figshare.29189858
```

After downloading, set an environment variable pointing to your dataset location:

```bash
export PHL_DATA_DIR=/path/to/openreact_chon_fh
```

Expected contents (edit to match your packaging):
- RTP (reactants / transition states / products)
- IRC (intrinsic reaction coordinate trajectories)
- NMS (normal-mode sampled geometries)

---

## Running the notebook(s)

Open and run the main notebook(s):

```bash
jupyter notebook
```

Suggested notebooks (rename to whatever you upload):
- `notebooks/01_train_phl_ani.ipynb`
- `notebooks/02_evaluate_models.ipynb`
- `notebooks/03_make_figures.ipynb` (optional)

---

## Method overview (PHL)

For a structure with coordinates \(\mathbf{R}\in\mathbb{R}^{3N}\), PHL trains the model energy \(\tilde{E}_\theta(\mathbf{R})\) with a combined objective:

\[
\mathcal{L} = \lambda_E\,\mathcal{L}_E + \lambda_F\,\mathcal{L}_F + \lambda_H\,\mathcal{L}_{\mathrm{HVP}}
\]

where the curvature term matches projected Hessians:

\[
\mathcal{L}_{\mathrm{HVP}} \propto \mathbb{E}_{v\sim\mathcal{N}(0,I)}\left\|\tilde{H}v - Hv\right\|^2.
\]

### Probing protocols

- **Randomized probes (default):** resample probe vectors each minibatch
- **Fixed probes (data-limited):** one probe vector per structure for the whole training

---

## Reproducing paper results

A typical reproduction workflow:

1. **Train models**
   - E–F baseline
   - E–F–HVP (PHL; Gaussian probes)
   - (optional) one-hot / column probing baseline
   - (optional) full Hessian supervision

2. **Evaluate**
   - Test set (interpolation)
   - IRC (pathway geometries)
   - NMS (extrapolative distortions)

3. **Generate figures**
   - RMSE summary plots (energies/forces/Hessians)
   - Bland–Altman plots (randomized vs fixed probe regimes)
   - timing plots

All outputs (checkpoints/metrics/figures) can be written to `results/` by default.

---

## HIPPYNN curvature training (HIP-NN models)

In addition to the ANI experiments, we implemented Hessian and HVP training functionality in **HIPPYNN** to enable curvature-supervised training of **HIP-NN** models. The upstream LANL HIPPYNN repository is available at:

```text
https://github.com/lanl/hippynn
```

An example script demonstrating the Hessian/HVP training workflow is provided here:

```text
hippynn/examples/hessian_training.py
```

---

## Citation

If you use this code or dataset in your work, please cite:

```bibtex
@article{rodriguez2026phl,
  title   = {Projected Hessian Learning (PHL): Fast Curvature Supervision for Accurate Machine-Learning Interatomic Potentials},
  author  = {Rodriguez, Austin and ...},
  journal = {Machine Learning: Science and Technology},
  year    = {2026},
  note    = {submitted / arXiv preprint}
}
```

---

## License

Add a `LICENSE` file (MIT or BSD-3-Clause recommended).

---

## Contact

- Austin Rodriguez (maintainer) — <email>
- Issues and pull requests are welcome.
