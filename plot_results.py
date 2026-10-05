"""Plot the STSP/MS-STSP portions of the manuscript experiments."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from geometry import evaluation_cloud
from protocol import CASES, DENTATE_P, RESULTS, ROOT

ART = ROOT / "output"
BLUE, RED = "#1559A6", "#C73632"
COLORS = {"STSP": BLUE, "MS-STSP": RED}
LABEL_NAMES = (
    "Astrocytes", "Cajal--Retzius", "Cck--Tox", "Endothelial",
    "GABA", "Granule immature", "Granule mature", "Microglia",
    "Mossy", "Neuroblast", "OL", "OPC", "Radial glia-like", "nIPC",
)


def configure():
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                         "font.size": 9, "pdf.fonttype": 42})
    ART.mkdir(parents=True, exist_ok=True)


def manifolds():
    fig = plt.figure(figsize=(9, 5.3))
    for i, (_, manifold, _, _, _) in enumerate(CASES[:6]):
        ax = fig.add_subplot(2, 3, i+1, projection="3d")
        points = evaluation_cloud(manifold, 6000, 1620900+i)
        if i < 4:
            ax.plot(*points.T, color=BLUE, lw=.7)
        else:
            ax.scatter(*points.T, s=.35, color=BLUE, alpha=.75,
                       linewidths=0, rasterized=True)
        ax.set_title(manifold.capitalize(), fontsize=9)
        ax.set_axis_off()
        ax.set_box_aspect(np.maximum(np.ptp(points, axis=0), .1))
        ax.view_init(18, -68)
    fig.tight_layout(pad=.2)
    fig.savefig(ART / "supp_manifolds.pdf", bbox_inches="tight")
    plt.close(fig)


def synthetic():
    data = pd.read_csv(RESULTS / "synthetic/summary.csv")
    for name, _, _, _, _ in CASES:
        rows = data.loc[data.case.eq(name)]
        if set(rows.method) != {"STSP", "MS-STSP"}:
            raise ValueError(f"Missing summary rows for {name}")
        fig, (scatter, bars) = plt.subplots(1, 2, figsize=(6.0, 2.6),
                                             gridspec_kw={"width_ratios": (1, 1.25)})
        for row in rows.itertuples():
            color = COLORS[row.method]
            scatter.errorbar(row.fit_mean, row.coverage_mean,
                             xerr=row.fit_sd, yerr=row.coverage_sd,
                             fmt="o", ms=5, capsize=2, color=color,
                             label=row.method)
            offset = 0 if row.method == "STSP" else 1
            bars.barh(offset+.18, row.fit_mean, height=.3, color=color,
                      label=r"$e_{\mathrm{fit}}$" if offset == 0 else None)
            bars.barh(offset-.18, row.coverage_mean, height=.3, color=color,
                      alpha=.45, label=r"$e_{\mathrm{cov}}$" if offset == 0 else None)
            bars.errorbar(row.fit_mean, offset+.18, xerr=row.fit_sd,
                          fmt="none", ecolor="#333333", capsize=1.5)
            bars.errorbar(row.coverage_mean, offset-.18, xerr=row.coverage_sd,
                          fmt="none", ecolor="#333333", capsize=1.5)
        scatter.set(xscale="log", yscale="log",
                    xlabel=r"$e_{\mathrm{fit}}$", ylabel=r"$e_{\mathrm{cov}}$")
        scatter.grid(color="#dddddd", lw=.4)
        scatter.legend(frameon=False, fontsize=8)
        bars.set_yticks([0, 1], ["STSP", "MS-STSP"])
        bars.set_xlabel("Mean error")
        bars.legend(frameon=False, fontsize=7, loc="lower right")
        fig.suptitle(name.replace("_", " "), fontsize=9)
        fig.tight_layout(pad=.6)
        fig.savefig(ART / f"errors_{name}_stsp_ms.pdf", bbox_inches="tight")
        plt.close(fig)


def fits():
    for name, _, _, _, _ in CASES:
        folder = RESULTS / "synthetic/fits" / name / "rep01"
        with np.load(folder / "selected_clouds.npz") as z:
            clouds = (z["stsp"], z["ms_stsp"])
        fig = plt.figure(figsize=(5.6, 2.8))
        for j, (label, y) in enumerate(zip(("STSP", "MS-STSP"), clouds)):
            ax = fig.add_subplot(1, 2, j+1, projection="3d")
            ax.scatter(*y.T, s=.15 if len(y) > 10000 else .3,
                       color=COLORS[label], alpha=.8, linewidths=0,
                       rasterized=True)
            ax.set_title(label, fontsize=9)
            ax.set_axis_off()
            ax.set_box_aspect(np.maximum(np.ptp(y, axis=0), .1))
            ax.view_init(18, -104)
        fig.subplots_adjust(left=.02, right=.98, bottom=.02, top=.92,
                            wspace=.03)
        fig.savefig(ART / f"fit_{name}_stsp_ms.pdf", bbox_inches="tight")
        plt.close(fig)


def diagnostics():
    root = RESULTS / "diagnostics"
    rate = pd.read_csv(root / "rate/all_repetitions.csv")
    summary = rate.groupby(["manifold", "sigma"]).fit_mean.agg(
        mean="mean", sd="std").reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.1))
    for ax, name in zip(axes, ("circle", "sphere")):
        rows = summary.loc[summary.manifold.eq(name)].sort_values("sigma")
        ax.errorbar(rows.sigma, rows["mean"], yerr=rows.sd, fmt="o-",
                    ms=3.7, color=BLUE, lw=1.2, capsize=1.7,
                    label=r"STSP: mean $\pm$ SD")
        high = rows.loc[rows.sigma.between(.08, .30)]
        scale = float(np.exp(np.log(high["mean"] / high.sigma**2).mean()))
        x = np.geomspace(.05, .35, 100)
        ax.plot(x, scale*x*x, "--", color="#333333", lw=1,
                label=r"Theoretical $O(\sigma^2)$ scaling")
        ax.set(xscale="log", yscale="log", xlabel=r"Noise level $\sigma$")
        ax.grid(color="#dddddd", lw=.4)
    axes[0].set_ylabel(r"Fitting error $e_{\mathrm{fit}}$")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(ART / "stsp_rate.pdf", bbox_inches="tight")
    plt.close(fig)

    conv = pd.read_csv(root / "convergence/all_repetitions.csv")
    summary = conv.groupby(["manifold", "iteration"]).fit_mean.agg(
        mean="mean", sd="std").reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.1))
    for ax, name in zip(axes, ("circle", "sphere")):
        rows = summary.loc[summary.manifold.eq(name)].sort_values("iteration")
        x, mean, sd = rows.iteration.to_numpy(), rows["mean"].to_numpy(), rows.sd.to_numpy()
        floor = float(mean[-11:].mean())
        guide = floor + (mean[0]-floor)*.16**x
        ax.fill_between(x, np.maximum(mean-sd, 1e-12), mean+sd,
                        color=BLUE, alpha=.18, linewidth=0)
        ax.semilogy(x, guide, "--", color="#333333", lw=1,
                    label="Theoretical geometric decay")
        ax.semilogy(x, mean, "o-", ms=3.5, color=BLUE, lw=1.2,
                    label=r"STSP: mean $\pm$ SD")
        ax.set(xlim=(0, 8), xlabel="Iteration $t$")
        ax.grid(color="#dddddd", lw=.4)
    axes[0].set_ylabel(r"Fitting error $e_{\mathrm{fit}}$")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(ART / "stsp_convergence.pdf", bbox_inches="tight")
    plt.close(fig)

    radius = pd.read_csv(root / "radius/all_repetitions.csv")
    summary = radius.groupby(["sigma", "h"]).fit_mean.agg(
        mean="mean", sd="std").reset_index()
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for sigma, color in ((.05, BLUE), (.20, RED)):
        rows = summary.loc[np.isclose(summary.sigma, sigma)].sort_values("h")
        ax.errorbar(rows.h, rows["mean"], yerr=rows.sd, fmt="o-",
                    ms=3.2, color=color, lw=1.1, capsize=1.5,
                    label=rf"$\sigma={sigma:.2f}$")
        winner = rows.iloc[rows["mean"].argmin()]
        ax.plot(winner.h, winner["mean"], "*", ms=8, color=color,
                markeredgecolor="white", markeredgewidth=.35)
    ax.set(xscale="log", yscale="log", xlabel="Neighbourhood radius $h$",
           ylabel=r"Fitting error $e_{\mathrm{fit}}$")
    ax.grid(color="#dddddd", lw=.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ART / "stsp_radius_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def standardize(x):
    x = x-x.mean(axis=0)
    return x/np.sqrt(np.mean(np.sum(x*x, axis=1)))


def align(x, target):
    a, b = target-target.mean(axis=0), x-x.mean(axis=0)
    u, _, vt = np.linalg.svd(b.T @ a)
    return b @ (u @ vt) + target.mean(axis=0)


def dentate():
    rows = []
    for p in DENTATE_P:
        folder = RESULTS / "dentate" / f"p{p}"
        data = pd.read_csv(folder / "selected.csv")
        if set(data.method) != {"STSP", "MS-STSP"}:
            raise ValueError(f"Missing selected methods at p={p}")
        rows.append(data)
    data = pd.concat(rows, ignore_index=True)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for method in ("STSP", "MS-STSP"):
        subset = data.loc[data.method.eq(method)].sort_values("p")
        ax.plot(subset.p, subset.average_sil, "o-", ms=4, lw=1.2,
                color=COLORS[method], label=method)
    ax.set_xticks(DENTATE_P)
    ax.set(xlabel="Retained principal components", ylabel="Average SIL")
    ax.grid(axis="y", color="#dddddd", lw=.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ART / "dentate_pca_sil_stsp_ms.pdf", bbox_inches="tight")
    plt.close(fig)

    folder = RESULTS / "dentate/p30"
    with np.load(folder / "input.npz") as z:
        x, labels = z["x"], z["labels"]
    with np.load(folder / "stsp.npz") as z:
        stsp_y = z["y"]
    with np.load(folder / "ms_stsp.npz") as z:
        ms_y = z["y"]
    params = dict(n_components=2, perplexity=80, early_exaggeration=24,
                  init="pca", learning_rate="auto", max_iter=1500,
                  random_state=20260927)
    embedded_input = TSNE(**params).fit_transform(standardize(x))
    embedded = [align(TSNE(**params).fit_transform(standardize(y)), embedded_input)
                for y in (stsp_y, ms_y)]
    palette = plt.get_cmap("tab20")
    colors = [palette(i) for i in (0,2,4,6,8,10,12,14,16,18,1,3,5,7)]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 4.0))
    for ax, y, title in zip(axes, embedded, ("STSP", "MS-STSP")):
        for j, label in enumerate(LABEL_NAMES):
            mask = labels == j
            ax.scatter(y[mask, 0], y[mask, 1], s=5, color=colors[j],
                       alpha=.82, linewidths=0, label=label)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    handles, names = axes[1].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=4, frameon=False,
               fontsize=7, markerscale=1.7)
    fig.subplots_adjust(left=.02, right=.98, top=.96, bottom=.25, wspace=.04)
    fig.savefig(ART / "dentate_tsne_stsp_ms.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("section", choices=("manifolds", "synthetic", "fits",
                                            "diagnostics", "dentate", "all"))
    args = parser.parse_args()
    configure()
    tasks = {"manifolds": manifolds, "synthetic": synthetic, "fits": fits,
             "diagnostics": diagnostics, "dentate": dentate}
    for name, task in tasks.items():
        if args.section in (name, "all"):
            task()


if __name__ == "__main__":
    main()
