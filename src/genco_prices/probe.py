"""Probes: can simple models read the AC-OPF nodal prices from GENCO's hidden state? (exploratory)

A probe maps the features of one bus in one scenario to the AC-OPF price of that bus (labels.py). Every probe sees
the bus identity (one-hot) and one feature set:
  - bus: nothing else (the probe learns the mean price of each bus);
  - inputs: what GENCO sees at the bus (load, voltage and reactive limits, bus type) and the generators at the bus
    (number, capacity, lowest and highest linear cost);
  - outputs: GENCO's predictions at the bus (Vm, Va, Pg, Qg);
  - emb: GENCO's final bus embedding (192 values, the input of the bus output head; scripts/predict_embeddings.py).
"ctx" adds scenario context: the scenario mean of each feature; for emb also the mean generator embedding; for
inputs also the one system price of the scenario (merit order without network, from the inputs only).

Rows (columns of results/per_scenario_probe_<tag>.csv; names in VARIANTS):
  - Direct probes predict the AC price. Ridge: ridge_bus, ridge_inputs, ridge_inputs_ctx, ridge_outputs,
    ridge_outputs_ctx, ridge_emb, ridge_emb_ctx. Small MLP: mlp_inputs_ctx, mlp_outputs_ctx, mlp_emb, mlp_emb_ctx.
  - Hybrids predict the AC price minus the price decoded from GENCO's dispatch (lam_genco), from a feature set with
    context plus the decoded price and its scenario mean. Hybrid price = decoded price + this correction:
    ridge_hybrid_bus, ridge_hybrid_outputs, ridge_hybrid_emb, mlp_hybrid_outputs, mlp_hybrid_emb.
  - Controls: the same per-bus correction (bus identity, base price and its scenario mean) on other base prices.
    ridge_dc_bus: on DC-OPF (lam_dc). ridge_learned_bus: on the decode with learned marginal generators
    (lam_learned). ridge_ref_bus: on the decode of the solver's dispatch (lam_ref); this is an oracle control,
    because the solver's dispatch is not available at inference. ridge_bus is also a control: no GENCO feature.
    The controls show how much of a gain comes from a learned per-bus correction of the DC price formula (for
    example losses) and not from GENCO's state.

Training data. Every probe and every correction fits its weights to the AC prices of the 800 tuning scenarios
(evaluate.split). DC-OPF uses no AC prices. The decode uses them only to choose two tolerances (lam_learned also
one threshold; its classifier learns from the solver's dispatch). So the probes get more help from AC prices than
DC-OPF and the decode. Penalty choice: GroupKFold over load groups on the tuning scenarios (ridge 5 folds, MLP 3).
Training targets are clipped to the 0.5 % and 99.5 % quantiles of the tuning targets; the cross-validation error
uses the true prices. The MLP (one hidden layer, 64 units) stops after 40 epochs at most; several fits reach this
cap. The report uses the report scenarios of results/per_scenario_<tag>.csv and the same buses as evaluate.report.

Seeds. Each MLP row runs with the seeds 0, 1 and 2 (weights, batch order, early-stopping split; the penalty is
chosen again for each seed). The column without a suffix is seed 0, <col>_seed1 and <col>_seed2 are the others.
The summary gives the range of the mean error over the seeds (seed_low, seed_high). Ridge has no seed.
Differences below INDICATIVE (0.15 per MWh, about the seed range of the MLP rows) are marked as indicative.

Threads. `fit` limits BLAS and OpenMP to one thread (threadpoolctl) and records the setting in
results/probe_<tag>.json. The result files come from one full `fit` run with OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1. Another BLAS library or CPU can change the MLP rows slightly.

Run from the repository root. `fit` needs results/genco_opf_case118_emb (scripts/predict_embeddings.py, GPU
recommended). It writes every row in one run: 19 min (1,144 s) and about 2 GB of memory on our laptop, one thread.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m genco_prices.probe fit --tag test3000
    uv run python -m genco_prices.probe summarize --tag test3000   # from results/per_scenario_probe_<tag>.csv only
    uv run python -m genco_prices.probe figure --tag test3000      # figures/fig4_probe.png
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_info, threadpool_limits

from .evaluate import METHODS, TIE, group_bootstrap, load_labels, split
from .paths import LABELS, PRED

# GENCO's final embeddings of the labelled scenarios (scripts/predict_embeddings.py). That run gives the same dispatch
# as PRED (graphkit evaluate) to within 0.052 MW (largest generator difference over the 3,000 labelled scenarios).
EMB = Path(os.environ.get("GENCO_EMB", "results/genco_opf_case118_emb"))
N_BUS = 118
CLIP_Q = 0.005
RIDGE_ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]  # per-sample penalty on standardised features
MLP_ALPHAS = [1e-3, 1e-1]
RIDGE_FOLDS, MLP_FOLDS = 5, 3
SEEDS = (0, 1, 2)  # MLP seeds; seed 0 is the main row
INDICATIVE = 0.15  # per MWh: about the seed range of the MLP rows; a smaller difference is indicative only
THREADS = 1  # BLAS and OpenMP threads in `fit`
EMB_COLS = [f"emb_{i:03d}" for i in range(192)]
IN_COLS = ["Pd", "Qd", "Vm_min", "Vm_max", "Qg_min", "Qg_max", "PQ", "PV", "REF",
           "n_gen", "cap", "pmin", "c1_min", "c1_max", "has_gen"]
OUT_COLS = ["Vm_pred", "Va_pred", "Pg_pred", "Qg_pred"]

VARIANTS = {  # column -> (probe, feature blocks, base price or None, name in the table); the probe predicts AC - base
    "ridge_bus": ("ridge", [], None, "Ridge, bus identity only"),
    "ridge_inputs": ("ridge", ["inputs"], None, "Ridge, inputs"),
    "ridge_inputs_ctx": ("ridge", ["inputs", "inputs_ctx"], None, "Ridge, inputs + context"),
    "ridge_outputs": ("ridge", ["outputs"], None, "Ridge, outputs"),
    "ridge_outputs_ctx": ("ridge", ["outputs", "outputs_ctx"], None, "Ridge, outputs + context"),
    "ridge_emb": ("ridge", ["emb"], None, "Ridge, embedding"),
    "ridge_emb_ctx": ("ridge", ["emb", "emb_ctx"], None, "Ridge, embedding + context"),
    "ridge_dc_bus": ("ridge", ["dc"], "lam_dc", "Ridge correction of DC-OPF, bus identity"),
    "ridge_hybrid_bus": ("ridge", ["decode"], "lam_genco", "Ridge hybrid, bus identity"),
    "ridge_hybrid_outputs": ("ridge", ["outputs", "outputs_ctx", "decode"], "lam_genco", "Ridge hybrid, outputs + context"),
    "ridge_hybrid_emb": ("ridge", ["emb", "emb_ctx", "decode"], "lam_genco", "Ridge hybrid, embedding + context"),
    "ridge_learned_bus": ("ridge", ["learned"], "lam_learned",
                          "Ridge correction of decode with learned marginal generators, bus identity"),
    "ridge_ref_bus": ("ridge", ["ref"], "lam_ref", "Ridge correction of decode on solver dispatch, bus identity (oracle)"),
    "mlp_inputs_ctx": ("mlp", ["inputs", "inputs_ctx"], None, "MLP, inputs + context"),
    "mlp_outputs_ctx": ("mlp", ["outputs", "outputs_ctx"], None, "MLP, outputs + context"),
    "mlp_emb": ("mlp", ["emb"], None, "MLP, embedding"),
    "mlp_emb_ctx": ("mlp", ["emb", "emb_ctx"], None, "MLP, embedding + context"),
    "mlp_hybrid_outputs": ("mlp", ["outputs", "outputs_ctx", "decode"], "lam_genco", "MLP hybrid, outputs + context"),
    "mlp_hybrid_emb": ("mlp", ["emb", "emb_ctx", "decode"], "lam_genco", "MLP hybrid, embedding + context"),
}
REFS = ["lam_dc", "lam_genco", "lam_learned", "lam_ref"]  # reference methods from results/per_scenario_<tag>.csv


def seed_name(col: str, seed: int) -> str:
    """Column of one seed: seed 0 keeps the variant name."""
    return col if seed == 0 else f"{col}_seed{seed}"


def seed_cols(per: pd.DataFrame, col: str) -> list[str]:
    """The columns of `col` for every seed that is in `per`, seed 0 first (one column for ridge and references)."""
    return [c for c in [seed_name(col, s) for s in SEEDS] if c in per.columns]


def rows(tag: str) -> tuple[pd.DataFrame, list[int], pd.DataFrame]:
    """Decode table of the tuning and report scenarios, with the buses that evaluate.report scores."""
    summ, _ = load_labels(tag)
    tune, _ = split(summ)
    per = pd.read_csv(f"results/per_scenario_{tag}.csv")
    dec = pd.read_parquet(LABELS / f"case118_{tag}_decode.parquet")
    dec = dec[dec.scenario.isin(tune + per.scenario.tolist())]
    dec = dec[dec[list(METHODS)].notna().all(axis=1)].sort_values(["scenario", "bus"]).reset_index(drop=True)
    dec["load_scenario_idx"] = dec.scenario.map(summ.set_index("scenario").load_scenario_idx)
    return dec, tune, per


def blocks(dec: pd.DataFrame) -> dict[str, np.ndarray]:
    """Feature blocks (float32), one row per row of `dec`."""
    ids = sorted(int(s) for s in dec.scenario.unique())
    flt = [("scenario", "in", ids)]
    key = dec[["scenario", "bus"]]
    codes = pd.factorize(dec.scenario, sort=True)[0]  # scenario position in `ids`

    def ctx(x: np.ndarray) -> np.ndarray:  # scenario mean of every column, broadcast to the rows
        cnt = np.bincount(codes, minlength=len(ids)).astype(np.float64)
        means = np.vstack([np.bincount(codes, weights=c, minlength=len(ids)) for c in x.T.astype(np.float64)]) / cnt
        return means.T[codes].astype(np.float32)

    bus = pd.read_parquet(PRED / "case118_ieee_predictions.parquet", filters=flt,
                          columns=["scenario", "bus"] + IN_COLS[:9] + OUT_COLS)
    gen = pd.read_parquet(PRED / "case118_ieee_gen_predictions.parquet", filters=flt)
    g = gen[gen.max_p_mw > 0].groupby(["scenario", "bus"]).agg(
        n_gen=("idx", "size"), cap=("max_p_mw", "sum"), pmin=("min_p_mw", "sum"),
        c1_min=("cp1_eur_per_mw", "min"), c1_max=("cp1_eur_per_mw", "max")).reset_index()
    b = key.merge(bus, how="left").merge(g, how="left")
    b["has_gen"] = b.n_gen.notna().astype(float)
    b = b.fillna({c: 0.0 for c in ["n_gen", "cap", "pmin", "c1_min", "c1_max"]})
    assert len(b) == len(dec) and b[IN_COLS + OUT_COLS].notna().all().all()
    out = {"onehot": np.eye(N_BUS, dtype=np.float32)[dec.bus.values]}
    out["inputs"] = b[IN_COLS].to_numpy(np.float32)
    out["inputs_ctx"] = np.c_[ctx(out["inputs"]), dec.lam_uniform.values].astype(np.float32)
    out["outputs"] = b[OUT_COLS].to_numpy(np.float32)
    out["outputs_ctx"] = ctx(out["outputs"])
    e = key.merge(pd.read_parquet(EMB / "case118_ieee_bus_embeddings.parquet", filters=flt), how="left")
    out["emb"] = e[EMB_COLS].to_numpy(np.float32)
    assert np.isfinite(out["emb"]).all()
    del e
    eg = pd.read_parquet(EMB / "case118_ieee_gen_embeddings.parquet", filters=flt).groupby("scenario")[EMB_COLS].mean()
    out["emb_ctx"] = np.c_[ctx(out["emb"]), eg.loc[ids].to_numpy(np.float32)[codes]]
    out["decode"] = np.c_[dec.lam_genco.values, ctx(dec.lam_genco.values[:, None])].astype(np.float32)
    for name, c in [("dc", "lam_dc"), ("learned", "lam_learned"), ("ref", "lam_ref")]:
        out[name] = np.c_[dec[c].values, ctx(dec[c].values[:, None])].astype(np.float32)
    return out


def design(bl: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    return np.concatenate([bl["onehot"]] + [bl[n] for n in names], axis=1)


class Scaler:
    def __init__(self, X: np.ndarray) -> None:
        self.mu = X.mean(0, dtype=np.float64)
        self.sd = X.std(0, dtype=np.float64)
        self.sd[self.sd == 0] = 1.0

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu.astype(np.float32)) / self.sd.astype(np.float32)


def ridge_fit(X: np.ndarray, y: np.ndarray, alphas: list[float]) -> tuple[Scaler, float, dict[float, np.ndarray]]:
    """Ridge on standardised features, one Gram matrix for all penalties (alpha per sample)."""
    sc, ym, n, p = Scaler(X), float(y.mean()), len(X), X.shape[1]
    G, r = np.zeros((p, p)), np.zeros(p)
    for i in range(0, n, 20_000):
        Z = sc(X[i:i + 20_000]).astype(np.float64)
        G += Z.T @ Z
        r += Z.T @ (y[i:i + 20_000] - ym)
    return sc, ym, {a: np.linalg.solve(G / n + a * np.eye(p), r / n) for a in alphas}


def mlp_fit(X: np.ndarray, y: np.ndarray, alpha: float, seed: int) -> tuple[Scaler, float, float, MLPRegressor]:
    sc, ym, ys = Scaler(X), float(y.mean()), float(y.std())
    m = MLPRegressor(hidden_layer_sizes=(64,), alpha=alpha, batch_size=512, learning_rate_init=1e-3, max_iter=40,
                     early_stopping=True, validation_fraction=0.1, n_iter_no_change=5, random_state=seed)
    with warnings.catch_warnings():  # the 40-epoch cap is part of the probe; `epochs` in the JSON records it
        warnings.simplefilter("ignore", ConvergenceWarning)
        m.fit(sc(X), (y - ym) / ys)
    return sc, ym, ys, m


def scenario_mae(pred: np.ndarray, true: np.ndarray, scen: np.ndarray) -> pd.Series:
    return pd.Series(np.abs(pred - true)).groupby(scen).mean()


def fit_variant(col: str, bl: dict[str, np.ndarray], dec: pd.DataFrame, tr: np.ndarray, te: np.ndarray,
                seed: int = 0) -> tuple:
    """Choose the penalty by GroupKFold on the tuning rows `tr`, refit on all of them, predict the rows `te`."""
    probe, names, base_col, _ = VARIANTS[col]
    X = design(bl, names)
    base = dec[base_col].values if base_col else np.zeros(len(dec))
    y_all = dec.lam_true.values - base
    lo, hi = np.quantile(y_all[tr], [CLIP_Q, 1 - CLIP_Q])
    y = np.clip(y_all, lo, hi)
    grp, scen, true = dec.load_scenario_idx.values, dec.scenario.values, dec.lam_true.values
    alphas = RIDGE_ALPHAS if probe == "ridge" else MLP_ALPHAS
    oof = {a: np.zeros(len(tr)) for a in alphas}
    for f_tr, f_te in GroupKFold(RIDGE_FOLDS if probe == "ridge" else MLP_FOLDS).split(tr, groups=grp[tr]):
        a_tr, a_te = tr[f_tr], tr[f_te]
        if probe == "ridge":
            sc, ym, ws = ridge_fit(X[a_tr], y[a_tr], alphas)
            Z = sc(X[a_te])
            for a in alphas:
                oof[a][f_te] = Z @ ws[a] + ym
        else:
            for a in alphas:
                sc, ym, ys, m = mlp_fit(X[a_tr], y[a_tr], a, seed)
                oof[a][f_te] = m.predict(sc(X[a_te])) * ys + ym
    cv = {a: float(scenario_mae(oof[a] + base[tr], true[tr], scen[tr]).mean()) for a in alphas}
    best = min(cv, key=cv.get)
    if probe == "ridge":
        sc, ym, ws = ridge_fit(X[tr], y[tr], [best])
        pred = sc(X[te]) @ ws[best] + ym
        info = {}
    else:
        sc, ym, ys, m = mlp_fit(X[tr], y[tr], best, seed)
        pred = m.predict(sc(X[te])) * ys + ym
        info = {"seed": seed, "epochs": int(m.n_iter_)}
    return pred + base[te], {"alpha": best, "cv_mae": cv, "features": int(X.shape[1]), "clip": [lo, hi], **info}


def tables(per: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Mean, load-group bootstrap CI, median, p95, win rate, paired differences against DC-OPF and the decode, and
    the range of the mean over the MLP seeds (seed_low, seed_high; empty for one column)."""
    boot = group_bootstrap(per, cols)
    i_dc, i_dec = cols.index("lam_dc"), cols.index("lam_genco")
    names = {**{c: METHODS[c] for c in REFS}, **{c: VARIANTS[c][3] for c in VARIANTS}}
    out = []
    for j, c in enumerate(cols):
        d_dc, d_dec = boot[:, j] - boot[:, i_dc], boot[:, j] - boot[:, i_dec]
        means = [per[s].mean() for s in seed_cols(per, c)]
        out.append({
            "method": c, "name": names[c], "mean": per[c].mean(),
            "ci_low": np.quantile(boot[:, j], 0.025), "ci_high": np.quantile(boot[:, j], 0.975),
            "median": per[c].median(), "p95": per[c].quantile(0.95),
            "seeds": len(means),
            "seed_low": min(means) if len(means) > 1 else np.nan,
            "seed_high": max(means) if len(means) > 1 else np.nan,
            "minus_dc": per[c].mean() - per.lam_dc.mean(),
            "minus_dc_low": np.quantile(d_dc, 0.025), "minus_dc_high": np.quantile(d_dc, 0.975),
            "minus_decode": per[c].mean() - per.lam_genco.mean(),
            "minus_decode_low": np.quantile(d_dec, 0.025), "minus_decode_high": np.quantile(d_dec, 0.975),
            "indicative_vs_dc": abs(per[c].mean() - per.lam_dc.mean()) < INDICATIVE and c != "lam_dc",
            "indicative_vs_decode": abs(per[c].mean() - per.lam_genco.mean()) < INDICATIVE and c != "lam_genco",
            "beats_dc": (per[c] < per.lam_dc - TIE).mean(), "beats_decode": (per[c] < per.lam_genco - TIE).mean(),
        })
    return pd.DataFrame(out).set_index("method")


def fit(tag: str, only: list[str] | None = None) -> None:
    """Fit every row (MLP: every seed) in one run and write all result files. `only`: fit some, print, write nothing."""
    t0 = time.perf_counter()
    with threadpool_limits(limits=THREADS):
        pools = [{k: p.get(k) for k in ("internal_api", "version", "num_threads")} for p in threadpool_info()]
        dec, tune, per_ref = rows(tag)
        report = per_ref.scenario.tolist()
        tr = np.flatnonzero(dec.scenario.isin(tune).values)
        te = np.flatnonzero(dec.scenario.isin(report).values)
        bl = blocks(dec)
        print(f"{len(tune)} tuning scenarios ({len(tr)} rows), {len(report)} report scenarios ({len(te)} rows); "
              f"features ready in {time.perf_counter() - t0:.0f} s; thread pools {pools}", flush=True)
        d_te = dec.iloc[te]
        per = per_ref[["scenario", "load_scenario_idx"] + REFS].set_index("scenario")
        for c in ["lam_dc", "lam_genco"]:  # the same buses as evaluate.report: the reference errors must agree
            check = scenario_mae(d_te[c].values, d_te.lam_true.values, d_te.scenario.values)
            assert np.allclose(check.loc[per.index].values, per[c].values, atol=1e-4), c
        info = {}
        for col in only or list(VARIANTS):
            for seed in SEEDS if VARIANTS[col][0] == "mlp" else (0,):
                t1, name = time.perf_counter(), seed_name(col, seed)
                pred, info[name] = fit_variant(col, bl, dec, tr, te, seed)
                per[name] = scenario_mae(pred, d_te.lam_true.values, d_te.scenario.values).loc[per.index].values
                print(f"{name}: alpha {info[name]['alpha']:g}, CV {info[name]['cv_mae']}, "
                      f"report mean {per[name].mean():.3f}, {time.perf_counter() - t1:.0f} s", flush=True)
    if only:
        print("--only: no result file written")
        return
    per.reset_index().round(5).to_csv(f"results/per_scenario_probe_{tag}.csv", index=False)
    meta = {"run": "one full fit, every row", "run_date": time.strftime("%d/%m/%Y"),
            "runtime_s": round(time.perf_counter() - t0), "thread_limit": THREADS,
            "env": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            "thread_pools": pools, "numpy": np.__version__, "sklearn": sklearn.__version__,
            "tuning_scenarios": len(tune), "tuning_rows": int(len(tr)), "report_scenarios": len(report),
            "report_rows": int(len(te)), "clip_quantile": CLIP_Q, "ridge_alphas": RIDGE_ALPHAS,
            "mlp_alphas": MLP_ALPHAS, "mlp_seeds": list(SEEDS), "ridge_folds": RIDGE_FOLDS, "mlp_folds": MLP_FOLDS,
            "indicative_below": INDICATIVE, "variants": info}
    Path(f"results/probe_{tag}.json").write_text(json.dumps(meta, indent=1))
    print(f"fit done in {time.perf_counter() - t0:.0f} s", flush=True)
    summarize(tag)


PAIRS = [  # (a, b): paired difference a minus b; does GENCO's embedding add over its outputs or the bus identity?
    ("ridge_emb", "ridge_outputs"), ("ridge_emb_ctx", "ridge_outputs_ctx"), ("ridge_emb_ctx", "ridge_inputs_ctx"),
    ("mlp_emb_ctx", "mlp_outputs_ctx"), ("mlp_emb_ctx", "mlp_inputs_ctx"),
    ("ridge_hybrid_emb", "ridge_hybrid_outputs"), ("ridge_hybrid_emb", "ridge_hybrid_bus"),
    ("mlp_hybrid_emb", "mlp_hybrid_outputs"), ("ridge_hybrid_bus", "ridge_dc_bus"), ("ridge_ref_bus", "ridge_dc_bus"),
    ("ridge_learned_bus", "ridge_dc_bus"),
]


def seed_range(per: pd.DataFrame, a: str, b: str) -> tuple[float, float]:
    """Range of mean(a) - mean(b) over the seeds (the same seed on both sides when both have seeds); NaN for none."""
    sa, sb = seed_cols(per, a), seed_cols(per, b)
    if len(sa) == 1 and len(sb) == 1:
        return np.nan, np.nan
    if len(sa) > 1 and len(sb) > 1:
        d = [per[x].mean() - per[y].mean() for x, y in zip(sa, sb)]
    else:
        d = [per[x].mean() - per[y].mean() for x in sa for y in sb]
    return min(d), max(d)


def pair_table(per: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Paired differences with load-group bootstrap CIs, seed range and the indicative mark."""
    cols = sorted({c for ab in pairs for c in ab})
    boot, idx = group_bootstrap(per, cols), {c: i for i, c in enumerate(cols)}
    out = []
    for a, b in pairs:
        d = boot[:, idx[a]] - boot[:, idx[b]]
        lo, hi = seed_range(per, a, b)
        diff = per[a].mean() - per[b].mean()
        out.append({"a": a, "b": b, "a_minus_b": diff, "ci_low": np.quantile(d, 0.025),
                    "ci_high": np.quantile(d, 0.975), "seed_low": lo, "seed_high": hi,
                    "indicative": abs(diff) < INDICATIVE, "a_better": (per[a] < per[b] - TIE).mean()})
    return pd.DataFrame(out)


def summarize(tag: str) -> None:
    per = pd.read_csv(f"results/per_scenario_probe_{tag}.csv")
    cols = REFS + [c for c in VARIANTS if c in per.columns]
    t = tables(per, cols)
    pd.set_option("display.width", 250)
    print(f"{len(per)} report scenarios, {per.load_scenario_idx.nunique()} load groups; MAE per scenario. "
          f"MLP rows: seed 0, seed_low/high = range of the mean over seeds {list(SEEDS)}. "
          f"indicative_*: difference below {INDICATIVE} per MWh")
    print(t.drop(columns="name").round(3).to_string())
    t.round(4).to_csv(f"results/probe_{tag}.csv")
    out = pair_table(per, [(a, b) for a, b in PAIRS if a in per.columns and b in per.columns])
    print("\npaired differences (load-group bootstrap, 10,000 draws; seed range for MLP rows):")
    print(out.round(3).to_string(index=False))
    out.round(4).to_csv(f"results/probe_pairs_{tag}.csv", index=False)


FIG_ROWS = [  # (column, label, colour); rows that are missing from the table are skipped
    ("lam_dc", "DC-OPF", "#2a78d6"),
    ("lam_genco", "Decode, GENCO dispatch", "#eb6834"),
    ("ridge_bus", "Probe: bus identity only", "#bdbcb6"),
    ("mlp_inputs_ctx", "Probe: GENCO inputs + context (MLP)", "#bdbcb6"),
    ("mlp_outputs_ctx", "Probe: GENCO outputs + context (MLP)", "#bdbcb6"),
    ("ridge_emb", "Probe: GENCO embedding (linear)", "#6250d6"),
    ("mlp_emb", "Probe: GENCO embedding (MLP)", "#6250d6"),
    ("ridge_hybrid_bus", "Decode + correction, bus identity", "#f4a582"),
    ("ridge_hybrid_outputs", "Decode + correction, GENCO outputs", "#f4a582"),
    ("ridge_hybrid_emb", "Decode + correction, GENCO embedding", "#f4a582"),
    ("ridge_dc_bus", "DC-OPF + correction, bus identity", "#86b6ef"),
]


def figure(tag: str) -> None:
    import matplotlib.pyplot as plt

    from .figures import GRID, INK, INK2
    from .paths import FIGURES

    t = pd.read_csv(f"results/probe_{tag}.csv", index_col=0)
    per = pd.read_csv(f"results/per_scenario_probe_{tag}.csv")
    rows = [r for r in FIG_ROWS if r[0] in t.index]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    y = np.arange(len(rows))[::-1]
    h = t.loc[[r[0] for r in rows]]
    a.barh(y, h["mean"], color=[r[2] for r in rows], height=0.62)
    a.errorbar(h["mean"], y, xerr=[h["mean"] - h.ci_low, h.ci_high - h["mean"]], fmt="none", ecolor=INK2, capsize=3,
               lw=1)
    for yi, c in zip(y, h.index):
        txt = f"{h.at[c, 'mean']:.2f}"
        if "seeds" in h.columns and h.at[c, "seeds"] > 1:  # MLP: range of the mean over the seeds
            txt += f" (seeds {h.at[c, 'seed_low']:.2f}-{h.at[c, 'seed_high']:.2f})"
        a.text(h.at[c, "ci_high"] + 0.08, yi, txt, va="center", fontsize=9, color=INK)
    a.axvline(t.loc["lam_dc", "mean"], color="#2a78d6", lw=1, ls="--", zorder=0)
    a.set_yticks(y, [r[1] for r in rows])
    a.set_xlabel("Mean price error (per MWh), 95 % CI")
    a.set_xlim(0, h.ci_high.max() * 1.4)
    a.grid(True, axis="x", color=GRID, lw=0.8)
    a.set_title(f"(a) Mean error, {len(per):,} report scenarios", loc="left", color=INK)
    for c, label, color, ls in [("lam_dc", "DC-OPF", "#2a78d6", "-"), ("ridge_dc_bus", "DC-OPF + correction", "#86b6ef", "-"),
                                ("lam_genco", "Decode, GENCO dispatch", "#eb6834", "-"),
                                ("ridge_hybrid_emb", "Decode + correction, embedding", "#f4a582", "-"),
                                ("mlp_emb", "Probe: embedding (MLP)", "#6250d6", "-")]:
        if c not in per.columns:
            continue
        x = np.sort(per[c].values)
        b.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=color, lw=2, ls=ls,
               label=f"{label} (median {np.median(x):.2f})")
    b.set_xscale("log")
    b.set_xlabel("Mean absolute price error in a scenario (per MWh, log scale)")
    b.set_ylabel("Share of scenarios")
    b.grid(True, color=GRID, lw=0.8)
    b.legend(frameon=False, loc="upper center", fontsize=8, ncol=2, bbox_to_anchor=(0.45, -0.17))
    b.set_title("(b) Error per scenario", loc="left", color=INK)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "fig4_probe.png", bbox_inches="tight")
    print("figure:", FIGURES / "fig4_probe.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fit", "summarize", "figure"])
    ap.add_argument("--tag", default="test3000")
    ap.add_argument("--only", nargs="*", choices=list(VARIANTS),
                    help="fit: fit these rows only and print them; no result file is written")
    args = ap.parse_args()
    if args.step == "fit":
        fit(args.tag, args.only)
    else:
        {"summarize": summarize, "figure": figure}[args.step](args.tag)


if __name__ == "__main__":
    main()
