"""Decode prices for all labelled scenarios and compare them with the AC-OPF labels.

Split: only scenarios whose AC-OPF solution agrees with the stored one. Load scenarios with
load_scenario_idx % 3 == 0 tune the decoder tolerances; the others are the report split.

    uv run python -m genco_prices.evaluate tune --tag test3000
    uv run python -m genco_prices.classify --tag test3000          # learned marginal generators (tuning third)
    uv run python -m genco_prices.evaluate decode --tag test3000
    uv run python -m genco_prices.evaluate report --tag test3000
    uv run python -m genco_prices.evaluate summarize --tag test3000   # from results/per_scenario_<tag>.csv only
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
from collections.abc import Iterator

import numpy as np
import pandas as pd

from . import classify
from .decode import active_set, copper_plate_price, prices
from .paths import LABELS, PRED, RAW, SCEN_PER_PARTITION

UNTUNED = {"margin_mw": 1.0, "bind_thr": 0.99}  # settings fixed before any tuning
TIE = 1e-3  # per MWh: a smaller difference between two scenario errors is a tie (e.g. the same active set as DC-OPF)
SCARCITY = 500.0  # per MWh: a scenario with an AC price at or above this is scored separately
METHODS = {  # column in the decode table -> name in the post
    "lam_uniform": "One system price",
    "lam_dc": "DC-OPF",
    "lam_genco": "Decode, GENCO dispatch",
    "lam_genco_untuned": "Decode, GENCO dispatch, untuned",
    "lam_vertex": "Decode, GENCO dispatch, vertex rule",
    "lam_learned": "Decode, GENCO dispatch, learned marginal generators",
    "lam_sub_br": "Decode, GENCO dispatch, solver's near-limit branches",
    "lam_sub_gen": "Decode, GENCO dispatch, solver's generator classes",
    "lam_ref": "Decode, solver dispatch",
}


def split(summ: pd.DataFrame) -> tuple[list[int], list[int]]:
    ok = summ[summ.agrees]
    tune = ok.scenario[ok.load_scenario_idx % 3 == 0]
    report = ok.scenario[ok.load_scenario_idx % 3 != 0]
    return sorted(int(s) for s in tune), sorted(int(s) for s in report)


def scenarios(ids: list[int]) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """(scenario, bus, gen, branch): GENCO prediction and stored solution per bus and generator, stored branches."""
    flt = [("scenario", "in", list(ids))]
    bus_g = dict(tuple(pd.read_parquet(PRED / "case118_ieee_predictions.parquet", filters=flt).groupby("scenario")))
    gen_g = dict(tuple(pd.read_parquet(PRED / "case118_ieee_gen_predictions.parquet", filters=flt).groupby("scenario")))
    for part in sorted({s // SCEN_PER_PARTITION for s in ids}):
        f = glob.glob(str(RAW / f"branch_data.parquet/scenario_partition={part}/*.parquet"))[0]
        br_g = dict(tuple(pd.read_parquet(f).groupby("scenario")))
        for s in sorted(s for s in ids if s // SCEN_PER_PARTITION == part):
            yield (s, bus_g[s].sort_values("bus").reset_index(drop=True),
                   gen_g[s].sort_values("idx").reset_index(drop=True), br_g[s].sort_values("idx"))


def load_labels(tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summ = pd.read_parquet(LABELS / f"case118_{tag}_summary.parquet")
    lmp = pd.read_parquet(LABELS / f"case118_{tag}_lmp.parquet").sort_values(["scenario", "bus"])
    return summ, lmp


def tune(tag: str) -> dict:
    summ, lmp = load_labels(tag)
    ids, _ = split(summ)
    lam = {s: g.lam_p.values for s, g in lmp[lmp.scenario.isin(ids)].groupby("scenario")}
    data = list(scenarios(ids))

    def mae(margin: float, thr: float, rule: str) -> float:
        err = [np.nanmean(np.abs(prices(gen, active_set(bus, gen, br, "pred", margin, thr, rule)) - lam[s]))
               for s, bus, gen, br in data]
        return float(np.mean(err))

    grid = [{"rule": "margin", "margin_mw": m, "bind_thr": t, "mae": mae(m, t, "margin")}
            for m, t in itertools.product([1, 3, 5, 7, 10, 20], [0.85, 0.9, 0.93, 0.95, 0.97, 0.99, 1.0, 1.01])]
    grid += [{"rule": "vertex", "margin_mw": np.nan, "bind_thr": t, "mae": mae(0, t, "vertex")}
             for t in [0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 1.0]]
    grid = pd.DataFrame(grid)
    print(f"tuning split: {len(ids)} scenarios")
    print(grid.round(3).to_string(index=False))
    best = {r: g.loc[g.mae.idxmin()] for r, g in grid.groupby("rule")}
    settings = {"margin_mw": float(best["margin"].margin_mw), "bind_thr": float(best["margin"].bind_thr),
                "vertex_bind_thr": float(best["vertex"].bind_thr), "tuning_scenarios": len(ids)}
    grid.to_csv(LABELS / f"case118_{tag}_tune_grid.csv", index=False)
    (LABELS / f"case118_{tag}_settings.json").write_text(json.dumps(settings, indent=1))
    print("chosen:", settings)
    return settings


def decode_all(tag: str) -> None:
    summ, lmp = load_labels(tag)
    st = json.loads((LABELS / f"case118_{tag}_settings.json").read_text())
    m, thr = st["margin_mw"], st["bind_thr"]
    clf = classify.load(tag)
    if clf is None:
        raise SystemExit("train the classifier first: python -m genco_prices.classify --tag " + tag)
    model, clf_thr = clf
    lab = dict(tuple(lmp.groupby("scenario")))
    rows, per = [], []
    for s, bus, gen, br in scenarios(sorted(int(x) for x in summ.scenario)):
        a_t = active_set(bus, gen, br, "target", m, thr, "margin")
        a_p = active_set(bus, gen, br, "pred", m, thr, "margin")
        a_u = active_set(bus, gen, br, "pred", UNTUNED["margin_mw"], UNTUNED["bind_thr"], "margin")
        a_v = active_set(bus, gen, br, "pred", 0.0, st["vertex_bind_thr"], "vertex")
        prob = model.predict_proba(classify.features(gen, a_p)[classify.X_COLS])[:, 1]
        a_l = classify.learned_set(gen, a_p, prob, clf_thr, m)
        lb = lab[s]
        rows.append(pd.DataFrame({
            "scenario": s, "bus": lb.bus.values, "lam_true": lb.lam_p.values, "lam_dc": lb.lam_p_dc.values,
            "lam_uniform": copper_plate_price(gen, bus.Pd.sum()),
            "lam_ref": prices(gen, a_t), "lam_genco": prices(gen, a_p), "lam_genco_untuned": prices(gen, a_u),
            "lam_vertex": prices(gen, a_v), "lam_learned": prices(gen, a_l),
            "lam_sub_gen": prices(gen, a_t.with_branches(a_p)),  # solver's generator classes, GENCO's branches
            "lam_sub_br": prices(gen, a_p.with_branches(a_t)),  # GENCO's generator classes, solver's branches
        }))
        per.append({"scenario": s, "n_marginal_ref": int(a_t.marginal.sum()), "n_marginal_genco": int(a_p.marginal.sum()),
                    "n_marginal_vertex": int(a_v.marginal.sum()), "n_marginal_learned": int(a_l.marginal.sum()),
                    "n_binding_ref": a_t.hb.shape[0],
                    "n_binding_genco": a_p.hb.shape[0]})
    pd.concat(rows).to_parquet(LABELS / f"case118_{tag}_decode.parquet")
    pd.DataFrame(per).merge(summ[["scenario", "load_scenario_idx", "agrees"]]).to_parquet(
        LABELS / f"case118_{tag}_active_sets.parquet")
    print(f"decoded {len(per)} scenarios with {st}")


def group_bootstrap(per: pd.DataFrame, cols: list[str], n: int = 10_000, seed: int = 0) -> np.ndarray:
    """Means of `cols` over resampled load groups (scenarios that share a load profile stay together)."""
    codes, uniq = pd.factorize(per.load_scenario_idx)
    G = len(uniq)
    sums = np.vstack([np.bincount(codes, weights=per[c].values, minlength=G) for c in cols])
    cnt = np.bincount(codes, minlength=G).astype(float)
    rng = np.random.default_rng(seed)
    out = np.empty((n, len(cols)))
    for i in range(n):
        w = np.bincount(rng.integers(0, G, G), minlength=G)
        out[i] = sums @ w / (cnt @ w)
    return out


def spread_r2(d: pd.DataFrame, col: str) -> float:
    """Pooled R2 of each bus price's deviation from its scenario mean (ignores the price level)."""
    dt = d.lam_true - d.groupby("scenario").lam_true.transform("mean")
    dm = d[col] - d.groupby("scenario")[col].transform("mean")
    return float(1 - ((dm - dt) ** 2).sum() / (dt**2).sum())


def summary_tables(per: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean, load-group bootstrap CI, median, 95th percentile, win rate against DC-OPF; paired differences to DC-OPF."""
    boot = group_bootstrap(per, cols)
    diff_cols = ["lam_genco", "lam_learned", "lam_vertex", "lam_ref"]
    pd_boot = boot[:, [cols.index(c) for c in diff_cols]] - boot[:, [cols.index("lam_dc")]]
    table = pd.DataFrame({
        "method": [METHODS[c] for c in cols],
        "mean": per[cols].mean().values,
        "ci_low": np.quantile(boot, 0.025, axis=0), "ci_high": np.quantile(boot, 0.975, axis=0),
        "median": per[cols].median().values, "p95": per[cols].quantile(0.95).values,
        "beats_dc": [(per[c] < per.lam_dc - TIE).mean() for c in cols],
        "ties_dc": [((per[c] - per.lam_dc).abs() <= TIE).mean() for c in cols],
    }, index=cols)
    diffs = pd.DataFrame({"mean_minus_dc": per[diff_cols].mean().values - per.lam_dc.mean(),
                          "ci_low": np.quantile(pd_boot, 0.025, axis=0), "ci_high": np.quantile(pd_boot, 0.975, axis=0)},
                         index=[METHODS[c] for c in diff_cols])
    return table, diffs


def summarize(tag: str) -> None:
    """The table of the post from results/per_scenario_<tag>.csv alone (no data download needed)."""
    per = pd.read_csv(f"results/per_scenario_{tag}.csv")
    table, diffs = summary_tables(per, list(METHODS))
    pd.set_option("display.width", 200)
    print(f"{len(per)} scenarios, {per.load_scenario_idx.nunique()} load groups")
    print(table.round(3).to_string())
    print(diffs.round(3).to_string())


def compare(tag: str, other: str) -> None:
    """Paired differences other minus tag (e.g. GENCO Base minus Small) on the same report scenarios."""
    a = pd.read_csv(f"results/per_scenario_{tag}.csv").set_index("scenario")
    b = pd.read_csv(f"results/per_scenario_{other}.csv").set_index("scenario")
    assert a.index.equals(b.index), "the two runs must share the report scenarios"
    for c in ["lam_genco", "lam_learned"]:
        d = pd.DataFrame({"x": b[c].values - a[c].values, "load_scenario_idx": a.load_scenario_idx.values})
        boot = group_bootstrap(d, ["x"])[:, 0]
        print(f"{METHODS[c]}: {other} minus {tag} {d.x.mean():+.3f} "
              f"[{np.quantile(boot, 0.025):+.3f}, {np.quantile(boot, 0.975):+.3f}]; "
              f"better {(d.x < -TIE).mean():.3f}, worse {(d.x > TIE).mean():.3f}")


def report(tag: str) -> None:
    summ, _ = load_labels(tag)
    _, ids = split(summ)
    dec = pd.read_parquet(LABELS / f"case118_{tag}_decode.parquet")
    dec = dec[dec.scenario.isin(ids)]
    dec = dec[dec[list(METHODS)].notna().all(axis=1)]  # buses outside the reference island have no price
    sets = pd.read_parquet(LABELS / f"case118_{tag}_active_sets.parquet").set_index("scenario")
    cols = list(METHODS)
    per = dec.groupby("scenario")[cols + ["lam_true"]].apply(
        lambda d: pd.Series({c: (d[c] - d.lam_true).abs().mean() for c in cols})).reset_index()
    per = per.merge(sets[["load_scenario_idx"]], left_on="scenario", right_index=True)
    lam_max = dec.groupby("scenario").lam_true.max()
    scarce = sorted(int(x) for x in lam_max[lam_max >= SCARCITY].index)
    per_all, per = per, per[~per.scenario.isin(scarce)]
    dec = dec[~dec.scenario.isin(scarce)]
    boot_all = group_bootstrap(per_all, ["lam_genco", "lam_dc"])
    all_diff = boot_all[:, 0] - boot_all[:, 1]
    boot_l = group_bootstrap(per, ["lam_learned", "lam_genco"])
    learned_diff = boot_l[:, 0] - boot_l[:, 1]

    table, diffs = summary_tables(per, cols)
    table["spread_r2"] = [spread_r2(dec, c) for c in cols]
    rs = sets.loc[per.scenario]
    counts = {
        "solved": int(len(summ)), "agree": int(summ.agrees.sum()), "report": int(len(per)),
        "report_load_groups": int(per.load_scenario_idx.nunique()),
        "marginal_generators_mean": {k: float(rs[f"n_marginal_{k}"].mean()) for k in ["ref", "genco", "learned", "vertex"]},
        "binding_branches_mean": {k: float(rs[f"n_binding_{k}"].mean()) for k in ["ref", "genco"]},
        "scarcity_scenarios": {str(s): {c: round(float(per_all.set_index("scenario").loc[s, c]), 2) for c in cols}
                               for s in scarce},
        "learned_minus_margin_rule": [round(float(per.lam_learned.mean() - per.lam_genco.mean()), 3),
                                      round(float(np.quantile(learned_diff, 0.025)), 3),
                                      round(float(np.quantile(learned_diff, 0.975)), 3)],
        "with_scarcity_genco_minus_dc": [round(float(per_all.lam_genco.mean() - per_all.lam_dc.mean()), 3),
                                         round(float(np.quantile(all_diff, 0.025)), 3),
                                         round(float(np.quantile(all_diff, 0.975)), 3)],
    }
    pd.set_option("display.width", 200)
    print(f"report split: {len(per)} scenarios, {counts['report_load_groups']} load groups; MAE per scenario; "
          f"{len(scarce)} scarcity scenarios (AC price >= {SCARCITY:.0f}) scored separately")
    print(table.round(3).to_string())
    print("\npaired difference to DC-OPF (load-group bootstrap, 10,000 draws):")
    print(diffs.round(3).to_string())
    print("\n", json.dumps(counts, indent=1))
    table.round(4).to_csv(f"results/headline_{tag}.csv")
    diffs.round(4).to_csv(f"results/paired_vs_dc_{tag}.csv")
    per.round(5).to_csv(f"results/per_scenario_{tag}.csv", index=False)
    (LABELS.parent / f"counts_{tag}.json").write_text(json.dumps(counts, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["tune", "decode", "report", "summarize", "compare"])
    ap.add_argument("--tag", default="test3000")
    ap.add_argument("--other", default="base", help="second run for `compare`")
    args = ap.parse_args()
    if args.step == "compare":
        compare(args.tag, args.other)
    else:
        {"tune": tune, "decode": decode_all, "report": report, "summarize": summarize}[args.step](args.tag)


if __name__ == "__main__":
    main()
