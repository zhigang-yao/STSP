"""Run the published STSP/MS-STSP synthetic fits with frozen settings."""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from protocol import CASES, FROZEN_SYNTHETIC, RESULTS
from stsp import iterate, weights


def metric(y, reference, reference_nn):
    fit = reference_nn.kneighbors(y, return_distance=True)[0][:, 0]
    coverage = (NearestNeighbors(n_neighbors=1, algorithm="kd_tree")
                .fit(y).kneighbors(reference, return_distance=True)[0][:, 0])
    return dict(fit_mean=float(fit.mean()),
                coverage_mean=float(coverage.mean()),
                joint_mean=float(max(fit.mean(), coverage.mean())))


def run_one(case, rep):
    name = case[0]
    source = RESULTS / "synthetic/inputs" / name / f"rep{rep:02d}.npz"
    folder = RESULTS / "synthetic/fits" / name / f"rep{rep:02d}"
    if (folder / "complete.json").exists():
        print(f"Already complete: {name} rep{rep:02d}")
        return
    if folder.exists() and any(folder.iterdir()):
        raise RuntimeError(f"Inspect incomplete output before rerun: {folder}")
    if not source.exists():
        raise FileNotFoundError(f"Generate this input first: {source}")
    with np.load(source) as z:
        noisy, reference = z["noisy"], z["reference"]
        sigma, d = float(z["sigma"]), int(z["d"])

    fixed = FROZEN_SYNTHETIC[name]
    ratio, st_t = fixed["st_h"][rep-1], fixed["st_t"][rep-1]
    triple, ms_t = fixed["ms_h"][rep-1], fixed["ms_t"][rep-1]
    stsp_y = iterate(noisy, sigma*ratio, d, st_t, sigma, stop=False).y
    hs = sigma*np.asarray(triple, float)
    branches = [iterate(noisy, h, d, ms_t, sigma, stop=False).y for h in hs]
    ms_y = np.tensordot(weights(hs), np.stack(branches), axes=(0, 0))

    reference_nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(reference)
    selected = pd.DataFrame([
        dict(method="STSP", ratio=ratio, h=sigma*ratio, iterations=st_t,
             scales="", **metric(stsp_y, reference, reference_nn)),
        dict(method="MS-STSP", ratio=np.nan, h=np.nan, iterations=ms_t,
             scales=";".join(f"{h:.12g}" for h in hs),
             **metric(ms_y, reference, reference_nn)),
    ])
    folder.mkdir(parents=True, exist_ok=True)
    selected.to_csv(folder / "selected.csv", index=False)
    np.savez_compressed(folder / "selected_clouds.npz", stsp=stsp_y, ms_stsp=ms_y)
    (folder / "complete.json").write_text(json.dumps(dict(
        case=name, replicate=rep, input=str(source), methods=["STSP", "MS-STSP"]
    ), indent=2))
    print(f"Completed: {name} rep{rep:02d}", flush=True)


def summarize(names):
    rows = []
    for name in names:
        for rep in range(1, 11):
            folder = RESULTS / "synthetic/fits" / name / f"rep{rep:02d}"
            if not (folder / "complete.json").exists():
                raise FileNotFoundError(f"Missing complete repetition: {folder}")
            selected = pd.read_csv(folder / "selected.csv")
            if set(selected.method) != {"STSP", "MS-STSP"}:
                raise ValueError(f"Unexpected methods: {folder}")
            selected["case"] = name
            selected["replicate"] = rep
            rows.append(selected)
    raw = pd.concat(rows, ignore_index=True)
    dest = RESULTS / "synthetic"
    raw.to_csv(dest / "selected_all.csv", index=False)
    summary = raw.groupby(["case", "method"], sort=False).agg(
        n=("replicate", "nunique"),
        fit_mean=("fit_mean", "mean"), fit_sd=("fit_mean", "std"),
        coverage_mean=("coverage_mean", "mean"),
        coverage_sd=("coverage_mean", "std"),
        joint_mean=("joint_mean", "mean"),
    ).reset_index()
    if not summary.n.eq(10).all():
        raise AssertionError("Every summary requires ten repetitions")
    summary.to_csv(dest / "summary.csv", index=False)
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[c[0] for c in CASES], nargs="*")
    parser.add_argument("--reps", type=int, nargs="*", default=list(range(1, 11)))
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if not args.reps or any(not 1 <= r <= 10 for r in args.reps):
        parser.error("Replicates must be in 1..10")
    names = [c[0] for c in CASES if args.case is None or c[0] in args.case]
    if args.summarize:
        summarize(names)
    else:
        for case in CASES:
            if case[0] in names:
                for rep in args.reps:
                    run_one(case, rep)


if __name__ == "__main__":
    main()
