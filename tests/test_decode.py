"""Checks of the decoder and of the label conversion on small cases with a known answer."""

import numpy as np
import pandas as pd

from genco_prices.decode import ActiveSet, copper_plate_price, fit_prices, prices, ptdf
from genco_prices.labels import convert


def test_one_binding_branch_two_marginal_generators():
    # LMP_n = lam - hb[0, n] * mu. Marginal costs 10 at bus 0 and 30 at bus 2 force lam = 10, mu = 20.
    hb = np.array([[0.0, -0.5, -1.0]])
    lam, mu = fit_prices(hb, gbus=np.array([0, 2]), mc=np.array([10.0, 30.0]), marginal=np.array([True, True]),
                         at_max=np.array([False, False]), at_min=np.array([False, False]))
    assert np.isclose(lam, 10.0) and np.isclose(mu[0], 20.0)
    np.testing.assert_allclose(lam - hb.T @ mu, [10.0, 20.0, 30.0])


def test_no_congestion_gives_one_price():
    lam, mu = fit_prices(np.zeros((0, 3)), gbus=np.array([0, 1]), mc=np.array([25.0, 25.0]),
                         marginal=np.array([True, True]), at_max=np.array([False, False]), at_min=np.array([False, False]))
    assert np.isclose(lam, 25.0) and mu.size == 0


def test_generator_at_pmax_bounds_the_price_from_below():
    # one marginal generator (mc 20) at bus 0; a cheaper generator at Pmax (mc 15) at bus 1: no congestion, price 20
    gen = pd.DataFrame({"bus": [0, 1]})
    a = ActiveSet(marginal=np.array([True, False]), at_max=np.array([False, True]), at_min=np.array([False, False]),
                  mc=np.array([20.0, 15.0]), hb=np.zeros((0, 2)), island=np.array([True, True]))
    np.testing.assert_allclose(prices(gen, a), [20.0, 20.0])


def test_ptdf_two_bus_line_carries_all_injection():
    # bus 0 = slack; 1 MW injected at bus 1 flows back over the single line 0-1: PTDF = -1 (flow measured 0 -> 1)
    br = pd.DataFrame({"from_bus": [0], "to_bus": [1], "x": [0.1], "tap": [0.0]})
    H, island = ptdf(br, 2, ref=0)
    assert island.all()
    np.testing.assert_allclose(H[0], [0.0, -1.0], atol=1e-9)


def test_copper_plate_price_single_quadratic_generator():
    gen = pd.DataFrame({"cp1_eur_per_mw": [10.0], "cp2_eur_per_mw2": [0.1], "min_p_mw": [0.0], "max_p_mw": [100.0]})
    assert np.isclose(copper_plate_price(gen, 50.0), 20.0, atol=1e-6)  # 10 + 2 * 0.1 * 50


def test_vertex_rule_keeps_k_plus_one_generators():
    from genco_prices.decode import active_set

    # 3-bus triangle, one branch above the threshold -> k = 1 -> two marginal generators: the two furthest from limits
    bus = pd.DataFrame({"bus": [0, 1, 2], "REF": [1, 0, 0], "Vm_pred": [1.0, 1.0, 1.0], "Va_pred": [0.0, -0.05, -0.10]})
    br = pd.DataFrame({"idx": [0, 1, 2], "from_bus": [0, 1, 0], "to_bus": [1, 2, 2], "br_status": [1, 1, 1],
                       "x": [0.1, 0.1, 0.1], "tap": [0.0, 0.0, 0.0], "rate_a": [50.0, 100.0, 1000.0]})
    y = 1 / 0.1j  # series admittance, no shunt
    br = br.assign(Yff_r=y.real, Yff_i=y.imag, Yft_r=-y.real, Yft_i=-y.imag, Ytf_r=-y.real, Ytf_i=-y.imag,
                   Ytt_r=y.real, Ytt_i=y.imag)
    gen = pd.DataFrame({"bus": [0, 1, 2], "p_mw_pred": [50.0, 48.0, 2.0], "min_p_mw": [0.0] * 3,
                        "max_p_mw": [100.0] * 3, "cp1_eur_per_mw": [10.0, 20.0, 30.0], "cp2_eur_per_mw2": [0.0] * 3})
    a = active_set(bus, gen, br, "pred", 0.0, 0.95, rule="vertex")
    assert a.hb.shape[0] == 1  # branch 0 carries about 50 MW on a 50 MVA rating; the others are below 60 %
    np.testing.assert_array_equal(a.marginal, [True, True, False])  # 50 and 48 MW from a limit, against 2 MW
    np.testing.assert_array_equal(a.at_min, [False, False, True])


def test_convert_keeps_transformer_outage_and_frees_reference_voltage():
    # bus 0 = reference; branch 1-2 has a tap ratio, so pandapower makes it a transformer; it is out of service
    bus = np.array([[1, 3, 0, 0, 0, 0, 1, 1.03, 0, 100, 1, 1.06, 0.94],
                    [2, 1, 50, 10, 0, 0, 1, 1.0, 0, 100, 1, 1.06, 0.94],
                    [3, 2, 20, 5, 0, 0, 1, 1.0, 0, 100, 1, 1.06, 0.94]], dtype=float)
    gen = np.array([[1, 0, 0, 100, -100, 1.03, 100, 1, 200, 0],
                    [3, 0, 0, 100, -100, 1.0, 100, 1, 200, 0]], dtype=float)
    branch = np.array([[1, 2, 0.01, 0.1, 0, 250, 250, 250, 0, 0, 1, -360, 360],
                       [2, 3, 0.01, 0.1, 0, 250, 250, 250, 1.02, 0, 0, -360, 360],
                       [1, 3, 0.01, 0.1, 0, 250, 250, 250, 0, 0, 1, -360, 360]], dtype=float)
    gencost = np.array([[2, 0, 0, 3, 0, 10, 0], [2, 0, 0, 3, 0, 20, 0]], dtype=float)
    net = convert({"version": "2", "baseMVA": 100.0, "bus": bus, "gen": gen, "branch": branch, "gencost": gencost})
    assert len(net.trafo) == 1 and not net.trafo.in_service.iloc[0]
    assert net.line.in_service.all()
    assert net.ext_grid.controllable.all()


def test_learned_set_keeps_one_marginal_and_the_active_mask():
    from genco_prices.classify import learned_set

    gen = pd.DataFrame({"bus": [0, 1, 2], "p_mw_pred": [90.0, 50.0, 5.0], "min_p_mw": [0.0] * 3, "max_p_mw": [100.0] * 3})
    a = ActiveSet(marginal=np.array([False, True, False]), at_max=np.array([True, False, False]),
                  at_min=np.array([False, False, False]), mc=np.array([10.0, 20.0, 30.0]), hb=np.zeros((0, 3)),
                  island=np.array([True, True, True]))  # generator 2 is not active (not in any class)
    b = learned_set(gen, a, prob=np.array([0.1, 0.2, 0.9]), thr=0.5, margin_mw=7.0)
    np.testing.assert_array_equal(b.marginal, [False, True, False])  # none above 0.5 among active: take the highest
    np.testing.assert_array_equal(b.at_max, [True, False, False])
    assert not b.at_min.any()


def test_summary_counts_ties_against_dc():
    from genco_prices.evaluate import summary_tables

    per = pd.DataFrame({c: [1.0, 2.0, 3.0, 4.0] for c in ["lam_uniform", "lam_dc", "lam_genco_untuned", "lam_vertex",
                                                          "lam_learned", "lam_sub_br", "lam_sub_gen", "lam_ref"]})
    per["lam_genco"] = [0.5, 2.0005, 3.5, 4.0]  # better, tie (< 0.001), worse, tie
    per["load_scenario_idx"] = [0, 0, 1, 1]
    table, _ = summary_tables(per, list(per.columns.drop("load_scenario_idx")))
    assert table.loc["lam_genco", "beats_dc"] == 0.25 and table.loc["lam_genco", "ties_dc"] == 0.5
