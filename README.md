# STSP and MS-STSP

This directory contains the Python implementation of STSP and MS-STSP and
the scripts for their numerical results in the final manuscript. All fits
call the same algorithm file, `stsp.py`. The published radii and iteration
counts are recorded in `protocol.py`. For the ten synthetic repetitions, the settings are
indexed by dataset and repetition, preserving the reported ten-repetition
results. MS-STSP uses exactly three radii.

The package does not include code for the six comparison methods. Its plots
show the STSP/MS-STSP parts of the paper figures; they are not replacements
for the complete eight-method figures in the manuscript. No script changes
the manuscript or the original results directory.

## Environment and a small example

Use Python 3.10 or newer with `numpy`, `scipy`, `pandas`, `scikit-learn`,
`matplotlib`, and `h5py`. For example:

```bash
cd /path/to/STSP/code_final1
python3 -m venv .venv
. .venv/bin/activate
python -m pip install numpy scipy pandas scikit-learn matplotlib h5py
```

The following self-contained example fits a noisy circle and writes no files:

```python
import numpy as np
from stsp import fit_stsp, fit_ms_stsp

rng = np.random.default_rng(1)
t = np.linspace(0, 2*np.pi, 300, endpoint=False)
clean = np.column_stack((np.cos(t), np.sin(t), np.zeros_like(t)))
noisy = clean + 0.08*rng.normal(size=clean.shape)
stsp = fit_stsp(noisy, h=0.20, d=1, tmax=8, s=0.08).y
ms_stsp = fit_ms_stsp(noisy, hs=[0.32, 0.40, 0.48],
                      d=1, tmax=5, s=0.08).y
print(stsp.shape, ms_stsp.shape)
```

## Files and paper outputs

| File | Purpose | Corresponding paper item |
| --- | --- | --- |
| `stsp.py` | Shared STSP/MS-STSP implementation. | All fitted results below. |
| `protocol.py` | Fixed final radii, iteration counts, dataset sizes, and settings. | Tables 2–4 and Figures 3–9. |
| `geometry.py` | Curves, sphere, area-uniform Torus, and evaluation-cloud samplers. | Synthetic data and Supplementary Figure S1. |
| `generate_data.py` | Generate the nine synthetic datasets, ten repetitions each. | Inputs for Tables 2–3, Figures 3–7, and Supplementary Figures S2–S13. |
| `run_synthetic.py` | Fit the synthetic datasets with fixed parameters and report fitting/coverage means and standard deviations. | STSP/MS-STSP rows of Tables 2–3 and Supplementary Tables S3–S4; fitted clouds for Figure 3 and Supplementary Figures S4–S13. |
| `run_diagnostics.py` | Run the three STSP simulation studies: error versus noise, iteration, and radius. | Figures 5–7. |
| `run_dentate.py` | Prepare and fit the mouse dentate gyrus single-cell RNA-seq dataset (2,930 cells, 14 annotated populations). | STSP/MS-STSP rows of Table 4 and data for Figures 8–9. |
| `plot_results.py` | Draw figures from saved outputs; it does not refit. | STSP/MS-STSP parts of Figures 3–9 and Supplementary Figures S1–S13. |

"Dentate gyrus" is the anatomical name of the brain tissue from which the
real single-cell dataset was obtained; `run_dentate.py` is the real-data
experiment, not another synthetic manifold or another algorithm.

Figures 1–2 are conceptual illustrations. Supplementary Table S1 is an
auxiliary theoretical calculation. Neither needs an STSP/MS-STSP fit runner.

## Run the experiments

```bash
# Synthetic data. The optional source path reuses the paper's exact
# evaluation clouds and the existing Torus Gaussian-noise arrays.
python generate_data.py --original-results ../code/Results
python run_synthetic.py
python run_synthetic.py --summarize

# STSP diagnostic figures.
python run_diagnostics.py --study all

# Mouse dentate gyrus data; provide the original count-matrix .h5ad file.
python run_dentate.py prepare --h5ad ../code/Data/single_cell/radius_screen_20260924/dentategyrus.h5ad
python run_dentate.py fit --p 20 30 40 50

# Draw figures after the corresponding result files exist.
python plot_results.py all
```

For one synthetic repetition, use `python generate_data.py --case
surface_torus --reps 1` followed by `python run_synthetic.py --case
surface_torus --reps 1`. Synthetic outputs are under `results/synthetic/`,
diagnostics under `results/diagnostics/`, real-data outputs under
`results/dentate/`, and plots under `output/`. These directories are created
on demand. Existing input files are not overwritten; completed fits are
skipped. Do not start duplicate runs in the same result directory.

The full-size experiments have not been rerun while assembling this compact
package. The implementation has been checked with small inputs; complete
paper-size runs need substantial time, memory, and disk space.
