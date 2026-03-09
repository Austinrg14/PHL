# Projected Hessian Learning (PHL)

Reference implementation of **Projected Hessian Learning (PHL)**, a fast, stochastic curvature-supervision method for training machine-learning interatomic potentials (MLIPs) using **Hessian-vector products (HVPs)** instead of full Hessian matrices.

PHL replaces explicit Hessian supervision with projected curvature targets of the form \(Hv\), enabling second-order training at force-like computational cost.

> **Paper:** *Projected Hessian Learning (PHL): Fast Curvature Supervision for Accurate Machine-Learning Interatomic Potentials* (arXiv / submitted to IOP **Machine Learning: Science and Technology (MLST)**)

---

## What's in this repository

- `PHL_training.ipynb`: end-to-end ANI training workflow with **E-F-HVP (PHL)** using Gaussian (Hutchinson-type) probing.
- `utils.py` and `__init__.py`: modified TorchANI source files required for Hessian-aware loading/batching.
- Checkpoint outputs written to `trainings/` during notebook training runs.

---

## Getting started

### Tested environment

The notebook and custom Hessian training workflow were tested with:

- **Python 3.12**
- **TorchANI 2.2.4**
- **PyTorch with CUDA 12.8**
- `wandb`, `numpy`, `tqdm`, `h5py`

Install TorchANI version:

```bash
pip install torchani==2.2.4
```

Install additional required Python package:

```bash
pip install h5py
```

Install the remaining notebook dependencies:

```bash
pip install wandb numpy tqdm notebook
```

> Important: custom Hessian training in this repository does **not** work with the current unmodified upstream TorchANI package.

### Required TorchANI source-file replacements (mandatory for Hessian training)

To enable Hessian labels and Hessian-aware batching, replace two files in your installed TorchANI package:

- `torchani/utils.py`
- `torchani/data/__init__.py`

Use the modified versions provided in this repository.

These replacements are required for:

- reading Hessian labels from the dataset
- correctly formatting Hessian tensors for autograd/HVP training

Find your installed TorchANI package path:

```bash
python -c "import pathlib, torchani; print(pathlib.Path(torchani.__file__).resolve().parent)"
```

Then copy the provided modified files into:

- `<TORCHANI_PACKAGE_DIR>/utils.py`
- `<TORCHANI_PACKAGE_DIR>/data/__init__.py`

Make sure these files are replaced **before running the training notebook**.

### Cloning the repository

```bash
git clone https://github.com/<USERNAME>/<PHL-REPO-NAME>.git
cd <PHL-REPO-NAME>
```

---

## Dataset access (OpenREACT)

This work uses **OpenREACT-CHON-EFH - Open REaction Dataset of Atomic ConfiguraTions comprising C, H, O, N with Energies, Forces, and Hessians**.  
The dataset can be downloaded from Figshare:

```text
https://doi.org/10.6084/m9.figshare.29189858
```

In the notebook, set:

```python
dspath = os.path.join(path, 'path/to/dataset.h5')
```

to point to your downloaded `.h5` file.

Expected contents (edit to match your packaging and local file layout):

- RTP (reactants / transition states / products)
- IRC (intrinsic reaction coordinate trajectories)
- NMS (normal-mode sampled geometries)

---

## Running the notebook

After completing the TorchANI file replacements above:

```bash
jupyter notebook PHL_training.ipynb
```

Before launching training:

- Set `dspath` in the notebook to your dataset `.h5` file.
- Update `wandb.init(..., entity="<USER>", ...)` with your W&B entity name.
- (Optional) change the W&B project name from `"PHL Training"` if needed.
- Adjust these training controls in the training cell as needed:
  - `training_runs`
  - `data_percentage`
  - `seed_number`

---

## Training outputs

Each training run in `PHL_training.ipynb` creates files in `trainings/`:

- `trainings/sae_training-<run>-hvp-<pct>pct.pt`
- `trainings/training_<run>_hvp_<pct>pct-latest.pt`
- `trainings/training_<run>_hvp_<pct>pct-best.pt`

Validation/test RMSE metrics are also logged to W&B for each run.

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
@article{rodriguez2026projected,
  title={Projected Hessian Learning: Fast Curvature Supervision for Accurate Machine-Learning Interatomic Potentials},
  author={Rodriguez, Austin and Smith, Justin S and Matin, Sakib and Lubbers, Nicholas and Barros, Kipton and Mendoza-Cortes, Jose L},
  journal={arXiv preprint arXiv:2603.04523},
  year={2026}
}
```

---

## License

Add a `LICENSE` file (MIT or BSD-3-Clause recommended).

---

## Contact

- Austin Rodriguez (maintainer) - <email>
- Issues and pull requests are welcome.
