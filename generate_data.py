"""Create shared noisy/evaluation clouds for the synthetic comparisons."""

import argparse
import json
from pathlib import Path

import numpy as np

from geometry import evaluation_cloud, sample
from protocol import CASES, RESULTS


def generate(case, rep, original_results=None, n_override=None, eval_override=None):
    case_name, manifold, n, sigma, d = case
    n = n_override or n
    neval = eval_override or (20000 if d == 1 else 30000)
    source_seed = 2026092300 + CASES.index(case)*100 + rep
    clean_seed = 2026100201 + 100*rep if manifold == "torus" else source_seed
    eval_seed = 2026100201 if manifold == "torus" else 2026092399 + CASES.index(case)
    clean = sample(manifold, n, clean_seed)
    ref = evaluation_cloud(manifold, neval, eval_seed)
    noise = sigma*np.random.default_rng(source_seed + 100000).normal(size=clean.shape)
    if original_results is not None:
        original_results = Path(original_results)
        if manifold != "torus" and n_override is None and eval_override is None:
            fixed = original_results / "frozen_current/inputs" / f"{case_name}.npz"
            if not fixed.exists():
                raise FileNotFoundError(f"Original evaluation cloud missing: {fixed}")
            with np.load(fixed) as z:
                ref = z["reference"].copy()
        if manifold == "torus" and n_override is None:
            old = (original_results / "independent_simulation_review_20260923"
                   / "inputs/surface_torus" / f"rep{rep:02d}/input.npz")
            if not old.exists():
                raise FileNotFoundError(f"Original Torus noise missing: {old}")
            with np.load(old) as z:
                noise = z["noisy"] - z["clean"]
    folder = RESULTS / "synthetic/inputs" / case_name
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"rep{rep:02d}.npz"
    if dest.exists():
        raise FileExistsError(f"Input already exists: {dest}")
    np.savez_compressed(dest, clean=clean, noisy=clean+noise,
                        reference=ref, sigma=sigma, d=d)
    (folder / f"rep{rep:02d}.json").write_text(json.dumps(dict(
        case=case_name, manifold=manifold, N=n, N_eval=len(ref), sigma=sigma,
        d=d, clean_seed=clean_seed, noise_seed=source_seed+100000,
        evaluation_seed=eval_seed,
        original_results=str(original_results) if original_results else None,
    ), indent=2))
    print(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[c[0] for c in CASES], nargs="*")
    parser.add_argument("--reps", type=int, nargs="*", default=list(range(1, 11)))
    parser.add_argument("--original-results", type=Path)
    parser.add_argument("--n", type=int, help="small-run override")
    parser.add_argument("--n-eval", type=int, help="small-run override")
    args = parser.parse_args()
    if not args.reps or any(not 1 <= r <= 10 for r in args.reps):
        parser.error("Replicates must be in 1..10")
    for case in CASES:
        if args.case is None or case[0] in args.case:
            for rep in args.reps:
                generate(case, rep, args.original_results, args.n, args.n_eval)


if __name__ == "__main__":
    main()
