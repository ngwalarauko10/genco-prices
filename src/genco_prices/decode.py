"""Decode nodal prices (LMPs) from a primal OPF solution: no training, one small LP per scenario.

DC optimality conditions:
  - a marginal generator g (strictly inside its limits) sets the price at its bus: LMP = c1_g + 2 c2_g Pg_g;
  - a near-binding branch l (|S| >= threshold * rate_a) gets a congestion price mu_l >= 0;
  - at every bus n: LMP_n = lam - sum_l mu_l * s_l * PTDF[l, n], with s_l the sign of the from-end flow.
`fit_prices` finds (lam, mu) with one LP; `active_set` reads the marginal generators and near-binding branches
from a dispatch. Two rules pick the marginal generators:
  - "margin": every generator more than `margin_mw` away from both limits;
  - "vertex": the k + 1 generators furthest from their limits, k = number of near-binding branches. With linear
    costs, a DC-OPF optimum is an LP vertex, where this count holds when the vertex is not degenerate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from scipy.sparse.csgraph import connected_components

from .paths import BASE_MVA

MAX_PRICE = 1e4  # bound on |lam| and mu (per MWh), far above any price in the data


def ptdf(br: pd.DataFrame, n_bus: int, ref: int) -> tuple[np.ndarray, np.ndarray]:
    """DC PTDF (in-service branches x buses) with the slack at `ref`, and the mask of buses in ref's island."""
    f, t = br.from_bus.values, br.to_bus.values
    tap = np.where(br.tap.values == 0, 1.0, br.tap.values)
    b = 1.0 / (br.x.values * tap)
    _, comp = connected_components(sparse.coo_matrix((np.ones(len(br)), (f, t)), shape=(n_bus, n_bus)), directed=False)
    island = comp == comp[ref]
    A = np.zeros((len(br), n_bus))
    A[np.arange(len(br)), f] = 1.0
    A[np.arange(len(br)), t] = -1.0
    B = A.T @ (b[:, None] * A)
    keep = island & (np.arange(n_bus) != ref)
    X = np.zeros((n_bus, n_bus))
    X[np.ix_(keep, keep)] = np.linalg.inv(B[np.ix_(keep, keep)])
    return (b[:, None] * A) @ X, island


def ac_loading(br: pd.DataFrame, vm: np.ndarray, va_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """|S| / rate_a (larger of the two ends) and the sign of the from-end active flow, from stored admittances."""
    V = vm * np.exp(1j * va_rad)
    f, t = br.from_bus.values, br.to_bus.values
    i_f = (br.Yff_r + 1j * br.Yff_i).values * V[f] + (br.Yft_r + 1j * br.Yft_i).values * V[t]
    i_t = (br.Ytf_r + 1j * br.Ytf_i).values * V[f] + (br.Ytt_r + 1j * br.Ytt_i).values * V[t]
    s_f, s_t = V[f] * np.conj(i_f), V[t] * np.conj(i_t)
    load = np.maximum(np.abs(s_f), np.abs(s_t)) * BASE_MVA / br.rate_a.values
    return load, np.sign(s_f.real)


@dataclass(frozen=True)
class ActiveSet:
    """What the decoder reads from one dispatch."""

    marginal: np.ndarray  # per generator
    at_max: np.ndarray
    at_min: np.ndarray
    mc: np.ndarray  # marginal cost per generator at this dispatch
    hb: np.ndarray  # signed PTDF rows of the near-binding branches (k x buses)
    island: np.ndarray  # buses connected to the reference bus

    def with_branches(self, other: "ActiveSet") -> "ActiveSet":
        """This set's generator classes with the other set's near-binding branches (same topology)."""
        return replace(self, hb=other.hb)


def active_set(bus: pd.DataFrame, gen: pd.DataFrame, br: pd.DataFrame, which: str,
               margin_mw: float = 5.0, bind_thr: float = 0.95, rule: str = "margin") -> ActiveSet:
    """Active set of the dispatch `which` ('pred' = GENCO, 'target' = stored solver solution)."""
    n = len(bus)
    ref = int(bus.index[bus.REF == 1][0]) if (bus.REF == 1).any() else 0
    on = br[br.br_status == 1]
    H, island = ptdf(on, n, ref)
    load, sgn = ac_loading(on, bus[f"Vm_{which}"].values, bus[f"Va_{which}"].values)
    binding = load >= bind_thr

    pg = gen[f"p_mw_{which}"].values
    pmin, pmax = gen.min_p_mw.values, gen.max_p_mw.values
    c1, c2 = gen.cp1_eur_per_mw.values, gen.cp2_eur_per_mw2.values
    active = (pmax > 0) & (c1 + c2 > 0) & island[gen.bus.values]
    slack = np.minimum(pg - pmin, pmax - pg)
    if rule == "margin":
        marginal = active & (slack > margin_mw)
        near_max = pg >= pmax - margin_mw
    elif rule == "vertex":
        k = min(int(binding.sum()) + 1, int(active.sum()))
        marginal = np.zeros(len(gen), dtype=bool)
        marginal[np.argsort(-np.where(active, slack, -np.inf), kind="stable")[:k]] = True
        near_max = pmax - pg <= pg - pmin
    else:
        raise ValueError(f"unknown rule {rule!r}")
    if not marginal.any():  # no generator inside its limits: take the one closest to the middle of its range
        mid = np.abs(pg - (pmin + pmax) / 2) / np.maximum(pmax - pmin, 1e-6)
        marginal = active & (mid == mid[active].min())
    at_max = active & ~marginal & near_max
    at_min = active & ~marginal & ~at_max
    return ActiveSet(marginal, at_max, at_min, 2 * c2 * pg + c1, sgn[binding][:, None] * H[binding], island)


def fit_prices(hb: np.ndarray, gbus: np.ndarray, mc: np.ndarray, marginal: np.ndarray, at_max: np.ndarray,
               at_min: np.ndarray, w_ineq: float = 1.0, w_mu: float = 1e-3) -> tuple[float, np.ndarray]:
    """(lam, mu) from the DC optimality conditions, as one LP (HiGHS).

    min  sum|r| + w_ineq * sum(s) + w_mu * sum(mu)
    s.t. LMP(bus_g) + r_g = mc_g          marginal generators (L1 fit)
         LMP(bus_g) + s_g >= mc_g         generators at Pmax (price at least their marginal cost)
         LMP(bus_g) - s_g <= mc_g         generators at Pmin (price at most their marginal cost)
         mu >= 0, s >= 0
    The inequalities and the small mu penalty are regularisation: they fix congestion prices that the marginal
    generators leave open (e.g. a nearly radial binding branch with no marginal generator behind it).
    """
    k = hb.shape[0]
    rows = {"m": np.where(marginal)[0], "hi": np.where(at_max)[0], "lo": np.where(at_min)[0]}
    m, nh, nl = len(rows["m"]), len(rows["hi"]), len(rows["lo"])
    nx = 1 + k + 2 * m + nh + nl  # x = [lam, mu (k), r+ (m), r- (m), s (nh + nl)]
    c = np.r_[0.0, np.full(k, w_mu), np.ones(2 * m), np.full(nh + nl, w_ineq)]

    def lmp_row(g: int) -> np.ndarray:
        row = np.zeros(nx)
        row[0], row[1:1 + k] = 1.0, -hb[:, gbus[g]]
        return row

    A_eq = np.array([lmp_row(g) for g in rows["m"]]).reshape(m, nx)
    for i in range(m):
        A_eq[i, 1 + k + i], A_eq[i, 1 + k + m + i] = 1.0, -1.0
    A_ub, b_ub = [], []
    for j, g in enumerate(rows["hi"]):  # -(LMP + s) <= -mc
        row = -lmp_row(g)
        row[1 + k + 2 * m + j] = -1.0
        A_ub.append(row), b_ub.append(-mc[g])
    for j, g in enumerate(rows["lo"]):  # LMP - s <= mc
        row = lmp_row(g)
        row[1 + k + 2 * m + nh + j] = -1.0
        A_ub.append(row), b_ub.append(mc[g])
    bounds = [(-MAX_PRICE, MAX_PRICE)] + [(0, MAX_PRICE)] * k + [(0, None)] * (2 * m + nh + nl)
    res = linprog(c, A_ub=np.array(A_ub) if A_ub else None, b_ub=np.array(b_ub) if b_ub else None,
                  A_eq=A_eq, b_eq=mc[rows["m"]], bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"price LP failed: {res.message}")
    return float(res.x[0]), res.x[1:1 + k]


def prices(gen: pd.DataFrame, a: ActiveSet) -> np.ndarray:
    """Nodal prices from an active set; NaN at buses outside the reference island."""
    lam, mu = fit_prices(a.hb, gen.bus.values, a.mc, a.marginal, a.at_max, a.at_min)
    return np.where(a.island, lam - a.hb.T @ mu, np.nan)


def copper_plate_price(gen: pd.DataFrame, demand_mw: float) -> float:
    """One system price: economic dispatch without network and losses (sum Pg = sum Pd), by bisection."""
    c1, c2 = gen.cp1_eur_per_mw.values, gen.cp2_eur_per_mw2.values
    pmin, pmax = gen.min_p_mw.values, gen.max_p_mw.values
    on = pmax > 0

    def supply(lam: float) -> float:
        p = np.where(c2[on] > 0, (lam - c1[on]) / np.maximum(2 * c2[on], 1e-12), np.where(lam >= c1[on], pmax[on], pmin[on]))
        return np.clip(p, pmin[on], pmax[on]).sum()

    lo, hi = 0.0, MAX_PRICE
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if supply(mid) < demand_mw else (lo, mid)
    return (lo + hi) / 2
