"""A learned rule for the marginal generators, from GENCO's dispatch (exploratory).

Binary target: is the generator marginal in the solver's dispatch (decoder's margin rule)? The fit uses these class
labels only; the decision threshold uses price labels: out-of-fold probabilities (5 folds over load groups) and the
price error on the tuning third, like the decoder tolerances. Everything uses the tuning third only. A generator that
is not marginal goes to its limit as in the margin rule (to the nearest limit if it is far from both), so this variant
differs from the margin rule only in the marginal mask. No prices are needed at inference.

    uv run python -m genco_prices.classify --tag test3000
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from .decode import ActiveSet, active_set, prices
from .paths import LABELS

X_COLS = ["slack_lo", "slack_hi", "rel", "range", "c1", "c1_rank", "n_bind_pred", "n_marg_pred", "gen"]
THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def features(gen: pd.DataFrame, a_p: ActiveSet) -> pd.DataFrame:
    """One row per generator: GENCO's dispatch relative to the limits, cost, and counts of GENCO's active set."""
    pg, pmin, pmax = gen.p_mw_pred.values, gen.min_p_mw.values, gen.max_p_mw.values
    c1 = gen.cp1_eur_per_mw.values
    active = a_p.marginal | a_p.at_max | a_p.at_min
    return pd.DataFrame({
        "gen": gen.idx.values, "active": active, "slack_lo": pg - pmin, "slack_hi": pmax - pg,
        "rel": (pg - pmin) / np.maximum(pmax - pmin, 1e-6), "range": pmax - pmin, "c1": c1,
        "c1_rank": pd.Series(np.where(active, c1, np.inf)).rank(method="average").values,
        "n_bind_pred": a_p.hb.shape[0], "n_marg_pred": int(a_p.marginal.sum()),
    })


def learned_set(gen: pd.DataFrame, a_p: ActiveSet, prob: np.ndarray, thr: float, margin_mw: float) -> ActiveSet:
    """GENCO's active set with the marginal generators from the classifier (at least one)."""
    pg, pmin, pmax = gen.p_mw_pred.values, gen.min_p_mw.values, gen.max_p_mw.values
    active = a_p.marginal | a_p.at_max | a_p.at_min
    marg = active & (prob >= thr)
    if not marg.any():
        marg = active & (prob == prob[active].max())
    near = np.minimum(pg - pmin, pmax - pg) <= margin_mw  # where the margin rule assigns a limit, use its rule
    near_max = np.where(near, pg >= pmax - margin_mw, pmax - pg <= pg - pmin)
    at_max = active & ~marg & near_max
    return replace(a_p, marginal=marg, at_max=at_max, at_min=active & ~marg & ~at_max)


def new_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0,
                                          categorical_features=[X_COLS.index("gen")])


def train(tag: str) -> None:
    from .evaluate import load_labels, scenarios, split

    summ, lmp = load_labels(tag)
    ids, _ = split(summ)
    st = json.loads((LABELS / f"case118_{tag}_settings.json").read_text())
    lam = {s: g.lam_p.values for s, g in lmp[lmp.scenario.isin(ids)].groupby("scenario")}
    group = summ.set_index("scenario").load_scenario_idx
    data, rows = [], []
    for s, bus, gen, br in scenarios(ids):
        a_p = active_set(bus, gen, br, "pred", st["margin_mw"], st["bind_thr"])
        a_t = active_set(bus, gen, br, "target", st["margin_mw"], st["bind_thr"])
        f = features(gen, a_p).assign(scenario=s, y=a_t.marginal)
        rows.append(f)
        data.append((s, gen, a_p, f.index.values))
    X = pd.concat(rows, ignore_index=True)
    fit_rows = X[X.active]
    oof = np.zeros(len(X))
    for tr, te in GroupKFold(5).split(fit_rows, fit_rows.y, fit_rows.scenario.map(group)):
        m = new_model().fit(fit_rows.iloc[tr][X_COLS], fit_rows.iloc[tr].y)
        oof[fit_rows.index[te]] = m.predict_proba(fit_rows.iloc[te][X_COLS])[:, 1]
    starts = np.cumsum([0] + [len(r) for r in rows[:-1]])
    curve = {}
    for thr in THRESHOLDS:
        err = [np.nanmean(np.abs(prices(gen, learned_set(gen, a_p, oof[o:o + len(gen)], thr, st["margin_mw"]))
                              - lam[s]))
               for (s, gen, a_p, _), o in zip(data, starts)]
        curve[thr] = float(np.mean(err))
    thr = min(curve, key=curve.get)
    model = new_model().fit(fit_rows[X_COLS], fit_rows.y)
    (LABELS / f"case118_{tag}_classifier.pkl").write_bytes(pickle.dumps(model))
    info = {"threshold": thr, "oof_price_mae_by_threshold": curve, "tuning_scenarios": len(ids)}
    (LABELS / f"case118_{tag}_classifier.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def load(tag: str) -> tuple[HistGradientBoostingClassifier, float] | None:
    p = LABELS / f"case118_{tag}_classifier.pkl"
    if not p.exists():
        return None
    return pickle.loads(p.read_bytes()), json.loads((LABELS / f"case118_{tag}_classifier.json").read_text())["threshold"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="test3000")
    train(ap.parse_args().tag)


if __name__ == "__main__":
    main()
