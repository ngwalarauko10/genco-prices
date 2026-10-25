"""Figures of the post.

    uv run python -m genco_prices.figures --tag test3000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

from .decode import ac_loading, active_set
from .evaluate import METHODS, load_labels, scenarios, split
from .paths import FIGURES, LABELS, PRED, RAW

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {"lam_uniform": "#8a8984", "lam_dc": "#2a78d6", "lam_genco": "#eb6834", "lam_ref": "#1baf7a",
       "lam_vertex": "#6250d6", "lam_sub_br": "#eb6834", "lam_sub_gen": "#eb6834"}
RING, EXTRA = "#0b0b0b", "#eb6834"
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.dpi": 200, "savefig.facecolor": SURF})


def bus_xy() -> np.ndarray:
    """IEEE 118 one-line layout from pandapower's case118 (bus i of the data = row i)."""
    import pandapower.networks as pn

    net = pn.case118()
    return np.array([json.loads(g)["coordinates"] for g in net.bus.geo])


def price_map(ax, xy, br, lam, gbus, extra, bind, norm, title):
    """Network coloured by nodal price; thick: near-binding branches; rings: marginal generators (red: extra)."""
    on = br[br.br_status == 1]
    seg = np.stack([xy[on.from_bus.values], xy[on.to_bus.values]], axis=1)
    ax.add_collection(LineCollection(seg[~bind], colors="#c9c8c2", linewidths=0.7, zorder=1))
    ax.add_collection(LineCollection(seg[bind], colors=INK, linewidths=2.4, zorder=2))
    ax.scatter(xy[:, 0], xy[:, 1], c=lam, cmap=BLUES, norm=norm, s=30, edgecolors="none", zorder=3)
    for b, color, lw in [(gbus[~extra], RING, 1.3), (gbus[extra], EXTRA, 2.8)]:
        ax.scatter(xy[b, 0], xy[b, 1], s=150, facecolors="none", edgecolors=color, linewidths=lw, zorder=4)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    ax.set_aspect("equal")
    ax.axis("off")


def fig_maps(tag: str, typical: int, failure: int) -> None:
    st = json.loads((LABELS / f"case118_{tag}_settings.json").read_text())
    dec = pd.read_parquet(LABELS / f"case118_{tag}_decode.parquet")
    per = pd.read_csv(f"results/per_scenario_{tag}.csv").set_index("scenario")
    xy = bus_xy()
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.6), gridspec_kw={"hspace": 0.08, "wspace": 0.02})
    data = {s: (bus, gen, br) for s, bus, gen, br in scenarios([typical, failure])}
    for row, (s, label) in enumerate([(typical, f"Typical, scenario {typical}"), (failure, f"Failure, scenario {failure}")]):
        bus, gen, br = data[s]
        d = dec[dec.scenario == s].sort_values("bus")
        norm = Normalize(vmin=min(d.lam_true.min(), d.lam_genco.min()), vmax=max(d.lam_true.max(), d.lam_genco.max()))
        sets = {w: active_set(bus, gen, br, w, st["margin_mw"], st["bind_thr"]) for w in ["target", "pred"]}
        for col, (which, lam_col, name) in enumerate([("target", "lam_true", "solver"),
                                                       ("pred", "lam_genco", "decoded from GENCO")]):
            load, _ = ac_loading(br[br.br_status == 1], bus[f"Vm_{which}"].values, bus[f"Va_{which}"].values)
            m = sets[which].marginal
            other = sets["pred" if which == "target" else "target"].marginal
            extra = (m & ~other)[m]  # marginal in this dispatch only
            err = "" if col == 0 else f" (error {per.loc[s, 'lam_genco']:.1f})"
            price_map(axes[row, col], xy, br, d[lam_col].values, gen.bus.values[m], extra,
                      load >= st["bind_thr"], norm, f"{label}: {name}{err}")
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=BLUES), ax=axes[row, :], shrink=0.75, pad=0.01)
        cb.set_label("Nodal price (per MWh)", color=INK)
        cb.outline.set_visible(False)
    handles = [plt.Line2D([], [], color=INK, lw=2.4, label="Branch at 95 % of its rating or more"),
               plt.Line2D([], [], marker="o", ls="", mfc="none", mec=RING, mew=1.3, ms=11, label="Marginal generator"),
               plt.Line2D([], [], marker="o", ls="", mfc="none", mec=EXTRA, mew=2.8, ms=11,
                          label="Marginal in this dispatch only")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.45, 0.02))
    fig.savefig(FIGURES / "fig1_price_maps.png", bbox_inches="tight")
    print("maps:", typical, failure)


def fig_summary(tag: str) -> None:
    per = pd.read_csv(f"results/per_scenario_{tag}.csv")
    head = pd.read_csv(f"results/headline_{tag}.csv", index_col=0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.1), gridspec_kw={"width_ratios": [1.05, 1]})
    for c in ["lam_uniform", "lam_dc", "lam_ref", "lam_genco"]:
        x = np.sort(per[c].values)
        a.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=COL[c], lw=2,
               ls="--" if c == "lam_uniform" else "-", label=f"{METHODS[c]} (median {np.median(x):.2f})")
    a.set_xscale("log")
    a.set_xlabel("Mean absolute price error in a scenario (per MWh, log scale)")
    a.set_ylabel("Share of scenarios")
    a.grid(True, color=GRID, lw=0.8)
    a.legend(frameon=False, loc="lower right", fontsize=8.5)
    a.set_title("(a) Error per scenario", loc="left", color=INK)

    rows = ["lam_dc", "lam_genco", "lam_learned", "lam_sub_br", "lam_sub_gen", "lam_ref"]
    names = ["DC-OPF", "Decode, GENCO dispatch", "GENCO, learned marginal generators", "GENCO, solver's near-limit branches",
             "GENCO, solver's generator classes", "Decode, solver dispatch"]
    colors = [COL["lam_dc"], COL["lam_genco"], "#f4a582", "#bdbcb6", "#bdbcb6", COL["lam_ref"]]  # grey: diagnostics
    h = head.loc[rows, ["mean", "ci_low", "ci_high"]]
    probe = Path(f"results/probe_{tag}.csv")
    if probe.exists():  # the best direct probe on GENCO's embeddings (probe.py)
        pr = pd.read_csv(probe).set_index("method").loc[["mlp_emb"], ["mean", "ci_low", "ci_high"]]
        h = pd.concat([h.iloc[:2], pr, h.iloc[2:]])
        names.insert(2, "Probe, GENCO embeddings (MLP)")
        colors.insert(2, COL["lam_vertex"])
    y = np.arange(len(h))[::-1]
    b.barh(y, h["mean"], color=colors, height=0.62)
    b.errorbar(h["mean"], y, xerr=[h["mean"] - h.ci_low, h.ci_high - h["mean"]], fmt="none", ecolor=INK2, capsize=3, lw=1)
    dc_mean = h.loc["lam_dc", "mean"]  # dashed reference line
    for yi, v, hi in zip(y, h["mean"], h.ci_high):
        b.text(max(hi, dc_mean) + 0.08, yi, f"{v:.2f}", va="center", fontsize=9, color=INK)
    b.axvline(dc_mean, color=COL["lam_dc"], lw=1, ls="--", zorder=0)
    b.set_yticks(y, names)
    b.set_xlabel("Mean price error (per MWh), 95 % CI")
    b.set_xlim(0, h.ci_high.max() * 1.25)
    b.grid(True, axis="x", color=GRID, lw=0.8)
    b.set_title("(b) Mean error by method", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_errors.png", bbox_inches="tight")


def pick_examples(tag: str) -> tuple[int, int]:
    """Typical = median GENCO error; failure = near the 95th percentile, where the solver's marginal generators
    remove most of the GENCO error (the mechanism of the post)."""
    per = pd.read_csv(f"results/per_scenario_{tag}.csv")
    typical = int(per.iloc[(per.lam_genco - per.lam_genco.median()).abs().argsort().iloc[0]].scenario)
    cand = per[(per.lam_sub_gen < 0.5 * per.lam_genco) & (per.lam_genco > 2 * per.lam_dc)]
    q95 = per.lam_genco.quantile(0.95)
    failure = int(cand.iloc[(cand.lam_genco - q95).abs().argsort().iloc[0]].scenario)
    return typical, failure


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="test3000")
    ap.add_argument("--typical", type=int)
    ap.add_argument("--failure", type=int)
    args = ap.parse_args()
    FIGURES.mkdir(exist_ok=True)
    fig_summary(args.tag)  # needs only results/*.csv
    if not (PRED.exists() and RAW.exists() and (LABELS / f"case118_{args.tag}_decode.parquet").exists()):
        print("Figure 1 needs the raw data, the GENCO predictions and the decode output (see reproduce.sh)")
        return
    t, f = pick_examples(args.tag)
    fig_maps(args.tag, args.typical or t, args.failure or f)


if __name__ == "__main__":
    main()
