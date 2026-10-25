"""Checks of the probe summary helpers (seed columns, seed range, indicative mark) on a small table."""

import numpy as np
import pandas as pd

from genco_prices.probe import INDICATIVE, pair_table, seed_cols, seed_name, seed_range, tables


def toy_per() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 60
    per = pd.DataFrame({"scenario": np.arange(n), "load_scenario_idx": np.arange(n) // 2,
                        "lam_dc": 2.0 + rng.random(n), "lam_genco": 3.0 + rng.random(n),
                        "lam_learned": 2.8 + rng.random(n), "lam_ref": 1.9 + rng.random(n),
                        "ridge_emb": 3.5 + rng.random(n), "ridge_outputs": 4.3 + rng.random(n)})
    for seed, shift in [(0, 0.0), (1, 0.1), (2, -0.05)]:
        per[seed_name("mlp_emb", seed)] = 3.4 + shift + rng.random(n)
        per[seed_name("mlp_emb_ctx", seed)] = 3.5 + shift + rng.random(n)
    return per


def test_seed_columns_and_range():
    per = toy_per()
    assert seed_name("mlp_emb", 0) == "mlp_emb" and seed_name("mlp_emb", 2) == "mlp_emb_seed2"
    assert seed_cols(per, "mlp_emb") == ["mlp_emb", "mlp_emb_seed1", "mlp_emb_seed2"]
    assert seed_cols(per, "ridge_emb") == ["ridge_emb"]
    lo, hi = seed_range(per, "mlp_emb_ctx", "mlp_emb")  # same seed on both sides
    d = [per[f"mlp_emb_ctx{s}"].mean() - per[f"mlp_emb{s}"].mean() for s in ["", "_seed1", "_seed2"]]
    assert np.isclose(lo, min(d)) and np.isclose(hi, max(d))
    assert np.isnan(seed_range(per, "ridge_emb", "ridge_outputs")[0])


def test_tables_and_pairs_mark_small_differences():
    per = toy_per()
    t = tables(per, ["lam_dc", "lam_genco", "lam_learned", "lam_ref", "ridge_emb", "mlp_emb"])
    means = [per[c].mean() for c in seed_cols(per, "mlp_emb")]
    assert t.at["mlp_emb", "seeds"] == 3 and np.isclose(t.at["mlp_emb", "seed_high"], max(means))
    assert t.at["ridge_emb", "seeds"] == 1 and np.isnan(t.at["ridge_emb", "seed_low"])
    assert not t.at["lam_dc", "indicative_vs_dc"]
    p = pair_table(per, [("ridge_emb", "ridge_outputs"), ("mlp_emb", "ridge_emb")]).set_index("a")
    for a in p.index:
        assert p.at[a, "indicative"] == (abs(p.at[a, "a_minus_b"]) < INDICATIVE)
        assert p.at[a, "ci_low"] <= p.at[a, "a_minus_b"] <= p.at[a, "ci_high"]
