"""AC-OPF price labels (LMPs) and DC-OPF baseline prices for datakit OPF scenarios (pandapower).

Rebuilds each stored scenario as a MATPOWER case, solves AC-OPF and DC-OPF, and compares the AC primal with the
stored PowerModels.jl solution. A scenario "agrees" when the cost is within 0.1 % and every net bus injection is
within 10 MW; only agreeing scenarios enter the evaluation.

    uv run python -m genco_prices.labels --scenarios results/test_ids_3000.txt --workers 1 --tag test3000
"""

from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.converter.pypower.from_ppc import from_ppc
from scipy import sparse

from .paths import BASE_MVA, LABELS, RAW, SCEN_PER_PARTITION

# pandapower's apparent-power OPF path (d2Sbr_dV2) still calls the sparse `.H` that SciPy removed.
for _cls in (sparse.csr_matrix, sparse.csc_matrix):
    if not hasattr(_cls, "H"):
        _cls.H = property(lambda self: self.conj().T)



def load_partition(partition: int) -> dict[str, pd.DataFrame]:
    out = {}
    for t in ["bus_data", "gen_data", "branch_data"]:
        files = glob.glob(str(RAW / f"{t}.parquet/scenario_partition={partition}/*.parquet"))
        out[t] = pd.read_parquet(files[0])
    return out


def build_ppc(bus: pd.DataFrame, gen: pd.DataFrame, br: pd.DataFrame) -> dict:
    """MATPOWER case (1-based bus ids) from one scenario's stored tables."""
    bus = bus.sort_values("bus")
    btype = np.where(bus.REF == 1, 3, np.where(bus.PV == 1, 2, 1))
    b = np.column_stack([
        bus.bus + 1, btype, bus.Pd, bus.Qd, bus.GS, bus.BS, np.ones(len(bus)),
        bus.Vm, bus.Va,  # stored Va is in degrees
        bus.vn_kv, np.ones(len(bus)), bus.max_vm_pu, bus.min_vm_pu,
    ])
    gen = gen.sort_values("idx")
    vm_at = dict(zip(bus.bus, bus.Vm))
    g = np.column_stack([
        gen.bus + 1, gen.p_mw, gen.q_mvar, gen.max_q_mvar, gen.min_q_mvar,
        gen.bus.map(vm_at), np.full(len(gen), BASE_MVA), gen.in_service,
        gen.max_p_mw, gen.min_p_mw,
    ])
    br = br.sort_values("idx")
    ratio = np.where((br.tap == 1.0) & (br.shift == 0.0), 0.0, br.tap)
    l = np.column_stack([
        br.from_bus + 1, br.to_bus + 1, br.r, br.x, br.b, br.rate_a, br.rate_a, br.rate_a,
        ratio, br["shift"], br.br_status, br.ang_min, br.ang_max,
    ])
    c = np.column_stack([
        np.full(len(gen), 2), np.zeros(len(gen)), np.zeros(len(gen)), np.full(len(gen), 3),
        gen.cp2_eur_per_mw2, gen.cp1_eur_per_mw, gen.cp0_eur,
    ])
    return {"version": "2", "baseMVA": BASE_MVA, "bus": b, "gen": g, "branch": l, "gencost": c}


def convert(ppc: dict) -> pp.pandapowerNet:
    """pandapower net from the MATPOWER case, with two repairs of `from_ppc` (pandapower 3.5.5):
    - transformers and impedances are created in service whatever BR_STATUS says: restore every branch status;
    - the reference generator becomes an ext_grid whose voltage the OPF fixes at its setpoint: make it controllable
      so |V| stays a variable within the bus limits, as in PowerModels (only the reference angle is fixed).
    The converter also drops the +-30 deg branch angle limits; they are inactive in these labels (max 21.8 deg).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = from_ppc(ppc, f_hz=60, validate_conversion=False)
    lk = net._from_ppc_lookups["branch"]
    for et in lk.element_type.unique():
        rows = lk[lk.element_type == et]
        net[et].loc[rows.element.astype(int).values, "in_service"] = ppc["branch"][rows.index.values, 10] == 1
    net.ext_grid["controllable"] = True
    return net


def solve(ppc: dict) -> pp.pandapowerNet:
    net = convert(ppc)
    # OPF_FLOW_LIM=0: limit apparent power |S| <= rate_a, as PowerModels does (pandapower default 2 = current)
    pp.runopp(net, init="pf", calculate_voltage_angles=True, verbose=False, OPF_FLOW_LIM=0)
    return net


def branch_loading(net: pp.pandapowerNet, br: pd.DataFrame) -> np.ndarray:
    """max(|S_from|, |S_to|) / rate_a for in-service branches, from the stored branch admittances."""
    br = br.sort_values("idx")
    br = br[br.br_status == 1]
    V = net.res_bus.vm_pu.values * np.exp(1j * np.radians(net.res_bus.va_degree.values))
    f, t = br.from_bus.values, br.to_bus.values
    i_f = (br.Yff_r + 1j * br.Yff_i).values * V[f] + (br.Yft_r + 1j * br.Yft_i).values * V[t]
    i_t = (br.Ytf_r + 1j * br.Ytf_i).values * V[f] + (br.Ytt_r + 1j * br.Ytt_i).values * V[t]
    s_max = np.maximum(np.abs(V[f] * np.conj(i_f)), np.abs(V[t] * np.conj(i_t))) * BASE_MVA
    return s_max / br.rate_a.values


def solve_dc(ppc: dict) -> pp.pandapowerNet:
    net = convert(ppc)
    pp.rundcopp(net, verbose=False)
    return net


def label_scenario(s: int, tabs: dict[str, pd.DataFrame]) -> tuple[dict, pd.DataFrame, pd.DataFrame] | None:
    """AC-OPF (labels) and DC-OPF (baseline) for one stored scenario."""
    bus = tabs["bus_data"][tabs["bus_data"].scenario == s]
    gen = tabs["gen_data"][tabs["gen_data"].scenario == s]
    br = tabs["branch_data"][tabs["branch_data"].scenario == s]
    ppc = build_ppc(bus, gen, br)
    t0 = time.perf_counter()
    try:
        net = solve(ppc)
    except Exception as e:  # noqa: BLE001 - report and continue
        print(f"scenario {s}: AC-OPF failed ({type(e).__name__})", flush=True)
        return None
    secs = time.perf_counter() - t0
    try:
        lam_dc = solve_dc(ppc).res_bus.lam_p.values
    except Exception as e:  # noqa: BLE001
        print(f"scenario {s}: DC-OPF failed ({type(e).__name__})", flush=True)
        lam_dc = np.full(len(bus), np.nan)
    busr = bus.sort_values("bus")
    pnet_err = np.abs(-net.res_bus.p_mw.values - (busr.Pg - busr.Pd).values).max()  # res_bus.p_mw is load convention
    lam = net.res_bus.lam_p.values
    cost_ref = (gen.cp2_eur_per_mw2 * gen.p_mw**2 + gen.cp1_eur_per_mw * gen.p_mw + gen.cp0_eur)[gen.in_service == 1].sum()
    loading = branch_loading(net, br)
    on = br.sort_values("idx")
    on = on[on.br_status == 1]
    summary = {
        "scenario": s, "load_scenario_idx": int(busr.load_scenario_idx.iloc[0]),
        "cost_ours": net.res_cost, "cost_ref": cost_ref,
        "vm_maxerr": np.abs(net.res_bus.vm_pu.values - busr.Vm.values).max(), "pnet_maxerr_mw": pnet_err,
        "lam_min": lam.min(), "lam_max": lam.max(), "lam_std": lam.std(),
        "n_binding": int((loading >= 0.99).sum()), "secs": secs,
    }
    lmp = pd.DataFrame({"scenario": s, "bus": busr.bus.values, "lam_p": lam, "lam_p_dc": lam_dc,
                        "vm": net.res_bus.vm_pu.values, "va_deg": net.res_bus.va_degree.values})
    branch = pd.DataFrame({"scenario": s, "idx": on.idx.values, "loading": loading})
    return summary, lmp, branch


def label_partition(job: tuple[int, list[int]]) -> list:
    partition, scen = job
    tabs = load_partition(partition)
    return [r for s in scen if (r := label_scenario(s, tabs)) is not None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", type=int, default=0)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--scenarios", type=Path, help="text file with one scenario id per line (overrides --partition/--n)")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes")
    ap.add_argument("--tag", default=None, help="output file tag (default: p<partition>)")
    args = ap.parse_args()

    if args.scenarios:
        ids = sorted(int(x) for x in args.scenarios.read_text().split())
    else:
        ids = sorted(load_partition(args.partition)["bus_data"].scenario.unique())[: args.n]
    jobs = sorted(pd.Series(ids).groupby(lambda i: ids[i] // SCEN_PER_PARTITION).apply(list).items())
    tag = args.tag or f"p{args.partition}"

    results = []
    t0 = time.perf_counter()
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            for k, res in enumerate(pool.imap_unordered(label_partition, jobs), 1):
                results += res
                print(f"{k}/{len(jobs)} partitions, {len(results)} scenarios, {time.perf_counter() - t0:.0f} s", flush=True)
    else:
        for job in jobs:
            results += label_partition(job)

    summary = pd.DataFrame([r[0] for r in results]).sort_values("scenario")
    summary["dcost_pct"] = 100 * (summary.cost_ours - summary.cost_ref) / summary.cost_ref
    summary["agrees"] = (summary.dcost_pct.abs() < 0.1) & (summary.pnet_maxerr_mw < 10)
    print(summary.describe().round(4).to_string())
    print(f"agrees with stored solution: {summary.agrees.mean():.1%} of {len(summary)}")
    out = LABELS
    out.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(out / f"case118_{tag}_summary.parquet")
    pd.concat([r[1] for r in results]).to_parquet(out / f"case118_{tag}_lmp.parquet")
    pd.concat([r[2] for r in results]).to_parquet(out / f"case118_{tag}_branch.parquet")


if __name__ == "__main__":
    main()
