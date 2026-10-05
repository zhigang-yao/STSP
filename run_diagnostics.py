"""Generate STSP rate, iteration and radius-sensitivity diagnostic tables."""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from geometry import evaluation_cloud, sample
from protocol import RESULTS
from stsp import iterate

N = 5000
TMAX = 30
SIGMAS = (.01, .02, .04, .08, .10, .15, .20, .25, .30, .35)


def fit_error(y, nearest):
    return float(nearest.kneighbors(y, return_distance=True)[0][:, 0].mean())


def one(study, manifold, rep):
    folder = RESULTS / "diagnostics" / study
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{manifold}_rep{rep:02d}.csv"
    if path.exists():
        return pd.read_csv(path)
    offset = 0 if manifold == "circle" else 1000
    seed = 2026100300 + offset + rep
    clean = sample(manifold, N, seed, sorted_u=(manifold == "circle"))
    noise = np.random.default_rng(seed+100000).normal(size=clean.shape)
    neval = 20000 if manifold == "circle" else 30000
    reference = evaluation_cloud(manifold, neval, 2026100399)
    nearest = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(reference)
    d = 1 if manifold == "circle" else 2
    rows = []
    if study == "rate":
        for sigma in SIGMAS:
            fit = iterate(clean+sigma*noise, 3*sigma, d, TMAX, sigma)
            rows.append(dict(manifold=manifold, replicate=rep, sigma=sigma,
                             h=3*sigma, iteration=fit.iterations,
                             fit_mean=fit_error(fit.y, nearest)))
    elif study == "convergence":
        sigma = .08
        fitted = iterate(clean+sigma*noise, .24, d, TMAX, sigma, stop=False)
        for t, y in enumerate((clean+sigma*noise, *fitted.history)):
            rows.append(dict(manifold=manifold, replicate=rep, sigma=sigma,
                             h=.24, iteration=t,
                             fit_mean=fit_error(y, nearest)))
    elif study == "radius":
        if manifold != "circle":
            raise ValueError("Radius study uses the circle")
        for sigma in (.05, .20):
            noisy = clean+sigma*noise
            for h in np.geomspace(.2*sigma, .5, 15):
                fitted = iterate(noisy, float(h), d, 10, sigma, stop=False)
                rows.append(dict(manifold=manifold, replicate=rep, sigma=sigma,
                                 h=float(h), iteration=10,
                                 fit_mean=fit_error(fitted.y, nearest)))
    else:
        raise ValueError(study)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    print(path, flush=True)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("rate", "convergence", "radius", "all"),
                        default="all")
    parser.add_argument("--reps", type=int, nargs="*", default=list(range(1, 11)))
    args = parser.parse_args()
    if not args.reps or any(not 1 <= r <= 10 for r in args.reps):
        parser.error("Replicates must be in 1..10")
    studies = ("rate", "convergence", "radius") if args.study == "all" else (args.study,)
    for study in studies:
        names = ("circle",) if study == "radius" else ("circle", "sphere")
        frames = [one(study, name, rep) for name in names for rep in args.reps]
        if len(args.reps) == 10:
            dest = RESULTS / "diagnostics" / study
            pd.concat(frames, ignore_index=True).to_csv(dest / "all_repetitions.csv",
                                                       index=False)
            (dest / "design.json").write_text(json.dumps(dict(
                study=study, N=N, repetitions=10, sampling="fixed reference",
                sigma=list(SIGMAS) if study == "rate" else (.08 if study == "convergence" else [.05, .20]),
            ), indent=2))


if __name__ == "__main__":
    main()
