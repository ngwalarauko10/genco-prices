"""Checks of the Core dual decoder on toy cases with a known answer."""

import numpy as np
import pandas as pd

from genco_prices.market import (REF, ZONES, day_bootstrap, decode_duals, ec_range_files, identifiable,
                                 other_mtus, predict_held_out)

ZONE_LIST = list(ZONES.values())


def toy_ptdf(rows: dict[int, dict[str, float]]) -> np.ndarray:
    P = np.zeros((len(rows), len(ZONE_LIST)))
    for i, r in rows.items():
        for z, v in r.items():
            P[i, ZONE_LIST.index(z)] = v
    return P


def test_lp_recovers_known_duals():
    # two constraints with independent PTDF differences; prices made with p = lam - mu @ P
    P = toy_ptdf({0: {"AT": -0.4, "HU": -0.3, "DE": 0.2}, 1: {"FR": 0.3, "BE": 0.1, "NL": -0.2}})
    lam, mu = 80.0, np.array([25.0, 60.0])
    lam_hat, mu_hat = decode_duals(P, lam - mu @ P, w_mu=1e-3)
    np.testing.assert_allclose(mu_hat, mu, atol=1e-6)
    np.testing.assert_allclose(lam_hat - mu_hat @ P, lam - mu @ P, atol=1e-6)


def test_hub_only_and_copied_rows_are_not_identifiable():
    i_ref = ZONE_LIST.index(REF)
    P = toy_ptdf({0: {"AT": -0.4, "DE": 0.2}, 1: {}, 2: {"AT": -0.4, "DE": 0.2}})  # row 1: hub only; row 2 = row 0
    rank, _ = identifiable(P)
    assert rank == 1 < P.shape[0]
    assert i_ref >= 0
    rank_ok, smin = identifiable(P[:1])
    assert rank_ok == 1 and smin > 1e-3


def test_held_out_zone_price_needs_the_right_rows():
    # the right rows predict every held-out zone price; rows of another constraint do not
    P = toy_ptdf({0: {"AT": -0.4, "HU": -0.3, "DE": 0.2}, 1: {"FR": 0.3, "BE": 0.1, "NL": -0.2}})
    p = 80.0 - np.array([25.0, 60.0]) @ P
    right = [predict_held_out(P, p, z, w_mu=1e-3)[0] for z in range(len(ZONE_LIST))]
    np.testing.assert_allclose(right, p, atol=1e-6)
    wrong_rows = toy_ptdf({0: {"CZ": 0.3, "SK": -0.2}})
    wrong = [predict_held_out(wrong_rows, p, z, w_mu=1e-3)[0] for z in range(len(ZONE_LIST))]
    assert np.abs(np.array(wrong) - p).max() > 1.0


def test_other_mtu_is_never_the_mtu_itself_and_is_repeatable():
    donors = list(pd.date_range("2026-10-14", periods=6, freq="15min", tz="Europe/Amsterdam"))
    mtus = pd.DatetimeIndex(donors + [donors[-1] + pd.Timedelta(minutes=15)])  # the last one has no donor rows
    a, b = other_mtus(mtus, donors), other_mtus(mtus, donors)
    assert a == b
    assert all(a[t] != t and a[t] in donors for t in mtus)


def test_day_bootstrap_keeps_whole_days():
    err = pd.Series([0.0, 0.0, 2.0, 2.0])
    day = pd.Series(["d1", "d1", "d2", "d2"])
    r = day_bootstrap(err, day, pd.Series([1, 2, 3, 4]), n_boot=200)
    assert r["mean_abs_error"] == 1.0 and r["share_within_1"] == 0.5 and r["days"] == 2
    lo, hi = r["mean_abs_error_ci95"]
    assert lo == 0.0 and hi == 2.0  # a draw holds whole days: only 0, 1 and 2 are possible means


def test_only_energy_charts_range_files_are_read(tmp_path):
    for name in ["AT_2026-10-02_2026-10-15.json", "AT_2026-10-15.json", "ATX_2026-10-02_2026-10-15.json"]:
        (tmp_path / name).write_text("{}")
    assert [f.name for f in ec_range_files("AT", tmp_path)] == ["AT_2026-10-02_2026-10-15.json"]
