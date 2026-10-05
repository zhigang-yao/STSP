"""Run the published STSP/MS-STSP fits on mouse Dentate Gyrus cells."""

import argparse
import json

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)

from protocol import DENTATE_P, FROZEN_DENTATE, RESULTS
from stsp import BatchProjector, Reference, weights

OUT = RESULTS / "dentate"


def h5_matrix(obj):
    import h5py
    if isinstance(obj, h5py.Dataset):
        return np.asarray(obj[...], float)
    shape = obj.attrs.get("h5sparse_shape", obj.attrs.get("shape"))
    return sparse.csr_matrix((obj["data"][...], obj["indices"][...],
                              obj["indptr"][...]), shape=shape).toarray()


def prepare(path):
    import h5py
    OUT.mkdir(parents=True, exist_ok=True)
    if any((OUT / f"p{p}/input.npz").exists() for p in DENTATE_P):
        raise FileExistsError("Some prepared inputs exist; inspect them before rerun")
    with h5py.File(path) as f:
        x = h5_matrix(f["X"])
        labels = np.asarray(f["obs"]["clusters"][...], int)
    if len(x) != 2930 or np.any(x < 0) or np.any(x.sum(axis=1) <= 0):
        raise ValueError("Unexpected Dentate count matrix")
    x = np.log1p(x / x.sum(axis=1)[:, None] * 1e4)
    variance = x.var(axis=0)
    selected_genes = np.argsort(variance)[-min(2000, np.count_nonzero(variance > 0)):]
    x = x[:, selected_genes]
    x = np.clip((x-x.mean(axis=0))/x.std(axis=0), -10, 10)
    scores = PCA(n_components=100, svd_solver="randomized",
                 random_state=20260925).fit_transform(x)
    labels = np.unique(labels, return_inverse=True)[1]
    for p in DENTATE_P:
        y = scores[:, :p].copy()
        y -= y.mean(axis=0)
        rms = float(np.sqrt(np.mean(np.sum(y*y, axis=1))))
        y /= rms
        folder = OUT / f"p{p}"
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "input.npz", x=y, labels=labels, rms=rms)
    (OUT / "preparation.json").write_text(json.dumps(dict(
        source=str(path), cells=len(x), classes=int(len(np.unique(labels))),
        genes=int(len(selected_genes)), p_values=DENTATE_P, pca_seed=20260925,
    ), indent=2))


def clustering(y, labels):
    k = len(np.unique(labels))
    records = []
    for seed in range(10):
        pred = KMeans(k, n_init=5, max_iter=500, init="k-means++",
                      random_state=seed).fit_predict(y)
        table = np.zeros((k, k), int)
        np.add.at(table, (labels, pred), 1)
        a, b = linear_sum_assignment(-table)
        records.append((table[a, b].sum()/len(labels),
                        adjusted_rand_score(labels, pred),
                        normalized_mutual_info_score(
                            labels, pred, average_method="geometric")))
    values = np.asarray(records)
    return dict(accuracy=float(values[:, 0].mean()),
                ari=float(values[:, 1].mean()), nmi=float(values[:, 2].mean()),
                accuracy_sd=float(values[:, 0].std(ddof=1)),
                ari_sd=float(values[:, 1].std(ddof=1)),
                nmi_sd=float(values[:, 2].std(ddof=1)))


def branch(x, projector, h, iterations):
    y = x.copy()
    for _ in range(iterations):
        y = projector.update(y, h)
    return y


def fit(p):
    folder = OUT / f"p{p}"
    if (folder / "complete.json").exists():
        print(f"Already complete: p={p}")
        return
    with np.load(folder / "input.npz") as z:
        x, labels = z["x"], z["labels"]
    projector = BatchProjector(Reference(x))
    st_h, st_t, ms_h, ms_t = FROZEN_DENTATE[p]
    stsp_y = branch(x, projector, st_h, st_t)
    hs = np.asarray(ms_h, float)
    ms_y = np.tensordot(weights(hs),
                        np.stack([branch(x, projector, float(h), ms_t) for h in hs]),
                        axes=(0, 0))
    rows = []
    for method, radii, t, y, filename in (
        ("STSP", (st_h,), st_t, stsp_y, "stsp.npz"),
        ("MS-STSP", hs, ms_t, ms_y, "ms_stsp.npz"),
    ):
        row = dict(method=method, scales=";".join(f"{h:.12g}" for h in radii),
                   iterations=t, average_sil=float(silhouette_score(y, labels)),
                   p=p, **clustering(y, labels))
        rows.append(row)
        np.savez_compressed(folder / filename, y=y, labels=labels)
    pd.DataFrame(rows).to_csv(folder / "selected.csv", index=False)
    (folder / "complete.json").write_text(json.dumps(dict(
        p=p, methods=["STSP", "MS-STSP"], scale_count=3,
    ), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--h5ad", required=True)
    run = sub.add_parser("fit")
    run.add_argument("--p", type=int, nargs="*", default=list(DENTATE_P))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.h5ad)
    else:
        for p in args.p:
            if p not in DENTATE_P:
                parser.error(f"Unsupported p={p}")
            fit(p)


if __name__ == "__main__":
    main()
