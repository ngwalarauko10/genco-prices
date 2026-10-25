"""Real data: read the duals of the European Core day-ahead market (flow-based) from prices and from the primal.

Core convention (zone-to-slack PTDFs, net positions NP in MW, shadow prices mu_c >= 0 in EUR/MWh):
    constraint c:  sum_z PTDF[c, z] * NP_z <= RAM_c
    zone price:    p_z = lam - sum_c mu_c * PTDF[c, z]
This is the formula of decode.py with zones in place of buses. Checks per 15-min market time unit (MTU):
  (A) identity: published shadow prices + PTDFs -> zonal price differences;
  (B) decode: zonal prices + the published active constraints -> mu (one LP, as decode.fit_prices). Where (A)
      holds, (B) only shows that the duals are identifiable. It is not separate evidence;
  (C) primal: published net positions + final domain -> the constraints at their limit -> mu as in (B);
  (D) leave one zone out: fit lam and mu on the other zones and predict the price of the held-out zone, once with
      the published active set and once with the active set of another MTU.
MTU classes, from the published net positions and the PTDF rows of the published constraints:
  at_limit: every published physical constraint has |RAM - flow| < 1 MW; off_limit: at least one has not;
  none: no published physical constraint. A physical constraint changes zonal price differences (not hub only).
PL: JAO publishes the PL allocation limits (limits on the PL net position) but not their shadow prices. When PL is
at one of these limits, the decode adds a row with a free dual for it. (A) and (D) do not score the PL price.

    uv run python -m genco_prices.market fetch                 # JAO + Energy-Charts downloads, cached under data/
    uv run python -m genco_prices.market analyse               # results/market_core_*, figures/fig3_core_duals.png
    uv run python -m genco_prices.market figure                # figure 3 again, from results/ only

Data: JAO publication tool (publicationtool.jao.eu, Core day-ahead) via jao-py. The zonal price differences come
from JAO's border price spreads (default). Energy-Charts zonal prices (api.energy-charts.info) are only a check: they
agree to 1e-13 EUR/MWh. The Energy-Charts prices of HR, RO and SK are "for private and internal use only": keep the
raw files local (data/ is not in git). Days with a clock change are not supported.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

TZ = "Europe/Amsterdam"
JAO_DIR = Path("data/jao")
EC_DIR = Path("data/prices_ec")
EC_URL = "https://api.energy-charts.info/price"
DAYS = [d.strftime("%Y-%m-%d") for d in pd.date_range("2026-10-02", "2026-10-15")]
DOMAIN_DAYS = ["2026-10-14", "2026-10-15"]  # (C) needs the final domain: 24 hourly requests per day
# Energy-Charts bidding zone -> JAO zone name (PTDF column ptdf_<JAO>)
ZONES = {"AT": "AT", "BE": "BE", "CZ": "CZ", "DE-LU": "DE", "FR": "FR", "HR": "HR", "HU": "HU", "NL": "NL",
         "PL": "PL", "RO": "RO", "SI": "SI", "SK": "SK"}
JAO_TABLES = ["active_constraints", "net_position", "allocationconstraint", "price_spread"]


# ----------------------------------------------------------------------------------------------------- fetch

def _get_json(session: requests.Session, url: str, params: dict, wait: float = 65.0, tries: int = 3) -> dict:
    """GET with a simple retry on HTTP 429 (rate limit)."""
    for _ in range(tries):
        r = session.get(url, params=params, timeout=120)
        if r.status_code != 429:
            r.raise_for_status()
            return r.json()
        time.sleep(wait)
    r.raise_for_status()
    return r.json()


def fetch_jao(days: list[str], pause: float = 1.0) -> list[str]:
    """Daily JAO tables (one request each). Returns the tables that failed (e.g. HTTP 500) as 'name day'."""
    from jao import JaoPublicationToolPandasClient

    JAO_DIR.mkdir(parents=True, exist_ok=True)
    c = JaoPublicationToolPandasClient()
    failed = []
    for day in days:
        d0 = pd.Timestamp(day, tz=TZ)
        d1 = d0 + pd.Timedelta(hours=23, minutes=59)
        query = {"active_constraints": lambda: c.query_active_constraints(d0),
                 "net_position": lambda: c.query_net_position(d0),
                 "allocationconstraint": lambda: c.query_allocationconstraint(d0, d1),
                 "price_spread": lambda: c.query_price_spread(d0, d1)}
        for name in JAO_TABLES:
            path = JAO_DIR / f"{name}_{day}.parquet"
            if path.exists():
                continue
            try:
                query[name]().to_parquet(path)
                print(f"saved {path}", flush=True)
            except Exception as e:  # noqa: BLE001 - JAO returns HTTP 500 for some days: skip them and say so
                failed.append(f"{name} {day}")
                print(f"FAILED {name} {day}: {type(e).__name__} {str(e)[:120]}", flush=True)
            time.sleep(pause)
    return failed


def hours(day: str) -> pd.DatetimeIndex:
    """The 24 local hours of `day`. Files use the local hour, so a day with a clock change is refused."""
    h = pd.date_range(pd.Timestamp(day, tz=TZ), pd.Timestamp(day, tz=TZ) + pd.Timedelta(days=1), freq="h",
                      inclusive="left")
    if len(h) != 24:
        raise ValueError(f"{day} has a clock change: not supported")
    return h


def fetch_domain(day: str, pause: float = 1.0) -> list[str]:
    """Final flow-based domain (all CNECs) for each hour of `day`; one file per hour. Sequential requests."""
    from jao import JaoPublicationToolClient
    from jao.parsers import parse_final_domain

    JAO_DIR.mkdir(parents=True, exist_ok=True)
    c = JaoPublicationToolClient()
    failed = []
    for mtu in hours(day):
        path = JAO_DIR / f"final_domain_{day}T{mtu.hour:02d}.parquet"
        if path.exists():
            continue
        try:
            rows = []
            for url, params, key in c.query_final_domain(mtu, urls_only=True):  # one call per 5,000 rows
                rows += _get_json(c.s, url, params)[key]
                time.sleep(pause)
            parse_final_domain(rows).to_parquet(path)
            print(f"saved {path} ({len(rows)} CNECs)", flush=True)
        except Exception as e:  # noqa: BLE001
            failed.append(f"final_domain {day}T{mtu.hour:02d}")
            print(f"FAILED final domain {mtu}: {type(e).__name__} {str(e)[:120]}", flush=True)
    return failed


def ec_path(zone: str, start: str, end: str) -> Path:
    return EC_DIR / f"{zone}_{start}_{end}.json"


def fetch_prices(start: str, end: str, wait: float = 65.0) -> None:
    """One Energy-Charts range request per zone (15-min prices); the API allows about one request per minute."""
    EC_DIR.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    first = True
    for zone in ZONES:
        path = ec_path(zone, start, end)
        if path.exists():
            continue
        if not first:
            time.sleep(wait)
        first = False
        t0, t1 = pd.Timestamp(start, tz=TZ), pd.Timestamp(end, tz=TZ) + pd.Timedelta(hours=23, minutes=45)
        params = {"bzn": zone, "start": t0.isoformat(), "end": t1.isoformat()}
        path.write_text(json.dumps(_get_json(s, EC_URL, params, wait=wait)))
        print(f"saved {path}", flush=True)


# ----------------------------------------------------------------------------------------------------- load

REF = "DE"  # price differences are taken against DE-LU
REL_MIN = 1.0  # EUR/MWh: relative errors only for published shadow prices above this
TOLS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]  # MW: (C) a CNEC is "at its limit" if RAM - flow <= tol
LIMIT_MW = 1.0  # a published constraint is "at its limit" if |RAM - flow| < LIMIT_MW
PRICE_TOL = 1.0  # EUR/MWh: a price (difference) "matches" if its error is at most PRICE_TOL
VIOLATION_MW = 1.0  # (C) an MTU is consistent if no CNEC of the final domain has a flow above RAM + VIOLATION_MW
TOL_C = 1.0  # MW: the tolerance of the headline numbers of (C); the sweep file has the others
N_BOOT, SEED = 2000, 0  # day-block bootstrap draws; seed of every random choice
CLASSES = ["at_limit", "off_limit", "none"]


def _read(name: str, day: str) -> pd.DataFrame | None:
    p = JAO_DIR / f"{name}_{day}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def ec_range_files(zone: str, directory: Path | None = None) -> list[Path]:
    """The Energy-Charts range files of `zone` (ZONE_start_end.json). Older single-day files are not read."""
    return sorted((directory or EC_DIR).glob(f"{zone}_????-??-??_????-??-??.json"))


def ec_prices(days: list[str]) -> pd.DataFrame:
    """Energy-Charts zonal prices (EUR/MWh), index = MTU start (local time), columns = JAO zone names."""
    out = {}
    for zone, jz in ZONES.items():
        parts = []
        for f in ec_range_files(zone):
            d = json.loads(f.read_text())
            idx = pd.to_datetime(d["unix_seconds"], unit="s", utc=True).tz_convert(TZ)
            parts.append(pd.Series(np.asarray(d["price"], dtype=float), index=idx))
        s = pd.concat(parts)  # ValueError if there is no range file
        out[jz] = s[~s.index.duplicated()].sort_index()
    df = pd.DataFrame(out)
    return df[df.index.strftime("%Y-%m-%d").isin(days)]


def jao_prices(days: list[str]) -> pd.DataFrame:
    """Zonal prices minus the DE-LU price, from JAO's border price spreads (border_X_Y = p_Y - p_X)."""
    ps = pd.concat([d for d in (_read("price_spread", day) for day in days) if d is not None])
    out = {REF: pd.Series(0.0, index=ps.index)}
    todo = True
    while todo:  # walk the border graph from DE outwards; fill a gap (NaN) through another border if one exists
        todo = False
        for col in ps.columns:
            a, b = col.removeprefix("border_").split("_")
            for x, y, sign in [(a, b, 1.0), (b, a, -1.0)]:
                if x not in out or y not in ZONES.values():
                    continue
                via = out[x] + sign * ps[col]
                if y not in out:
                    out[y], todo = via, True
                elif out[y].isna().any() and via[out[y].isna()].notna().any():
                    out[y], todo = out[y].fillna(via), True
    return pd.DataFrame(out)[list(ZONES.values())]


def active(days: list[str]) -> pd.DataFrame:
    """Published active constraints (one row per MTU and CNEC), with a CNEC key and an 'hub only' flag."""
    ac = pd.concat([d for d in (_read("active_constraints", day) for day in days) if d is not None]).reset_index()
    ac["key"] = cnec_key(ac.cnec_name, ac.cont_name, ac.direction)
    d = ac[[f"ptdf_{z}" for z in ZONES.values()]].to_numpy(float)
    ac["hub_only"] = np.abs(d - d[:, [list(ZONES.values()).index(REF)]]).max(axis=1) < 1e-6
    return ac


def cnec_key(name: pd.Series, cont: pd.Series, direction: pd.Series) -> pd.Series:
    return name.fillna("").str.strip() + " | " + cont.fillna("").str.strip() + " | " + direction.fillna("")


def pl_limits(days: list[str], tol: float = 1.0) -> pd.DataFrame:
    """PL net position against its two published allocation limits, per MTU.

    at_export: NP_PL = PL_export limit (within `tol` MW); at_import: -NP_PL = PL_import limit (within `tol` MW).
    """
    parts = []
    for day in days:
        al, npos = _read("allocationconstraint", day), _read("net_position", day)
        if al is None or npos is None:
            continue
        x = npos.PL.reindex(al.index).astype(float)
        parts.append(pd.DataFrame({"at_export": (al.PL_export.astype(float) - x).abs() <= tol,
                                   "at_import": (al.PL_import.astype(float) + x).abs() <= tol}, index=al.index))
    return pd.concat(parts) if parts else pd.DataFrame(columns=["at_export", "at_import"], dtype=bool)


def allocation_rows(pl: pd.DataFrame) -> pd.DataFrame:
    """The PL allocation constraints at their limit as extra rows (input: pl_limits).

    JAO publishes the limits but not their shadow prices, and they are not in the active-constraint list.
    Row: ptdf_PL = +1 (export: NP_PL <= limit) or -1 (import: -NP_PL <= limit); the shadow price is unknown (NaN).
    """
    rows = []
    for name, sign, col in [("PL_export", 1.0, "at_export"), ("PL_import", -1.0, "at_import")]:
        for t in pl.index[pl[col].to_numpy(bool)]:
            r = {"mtu": t, "tso": "PSE", "cnec_name": f"Allocation constraint {name}", "cont_name": "",
                 "direction": "", "shadow_price": np.nan, "hub_only": False}
            r |= {f"ptdf_{z}": 0.0 for z in ZONES.values()} | {"ptdf_PL": sign}
            rows.append(r)
    cols = ["mtu", "tso", "cnec_name", "cont_name", "direction", "shadow_price", "hub_only"]
    out = pd.DataFrame(rows, columns=cols + [f"ptdf_{z}" for z in ZONES.values()])
    out["key"] = cnec_key(out.cnec_name, out.cont_name, out.direction)
    return out


def published_margins(ac: pd.DataFrame, days: list[str]) -> pd.DataFrame:
    """RAM minus flow of each published active constraint, with its own PTDF row and the published net positions.

    A constraint with a positive shadow price must be at its limit (margin 0). This needs no final domain.
    ram_mcp is JAO's own margin after market coupling; it agrees with `margin` to within 1 MW here.
    """
    npos = pd.concat([d for d in (_read("net_position", day) for day in days) if d is not None])
    npos = npos.rename(columns={"DE_LU": "DE"})
    nodes = [c[5:] for c in ac.columns if c.startswith("ptdf_") and c[5:] in npos.columns]
    x = npos.reindex(ac.mtu)[nodes].to_numpy(float)
    flow = (ac[[f"ptdf_{z}" for z in nodes]].fillna(0.0).to_numpy(float) * x).sum(axis=1)
    cols = ["mtu", "key", "tso", "hub_only", "shadow_price", "ram", "ram_mcp"]
    return ac[cols].assign(margin=ac.ram.to_numpy(float) - flow)


def mtu_classes(pm: pd.DataFrame, mtus: pd.Index) -> pd.Series:
    """'at_limit', 'off_limit' or 'none' per MTU (see the module docstring); pm: published_margins."""
    phys = pm[~pm.hub_only.astype(bool)]
    at = phys.groupby("mtu").margin.apply(lambda v: bool((v.abs() < LIMIT_MW).all()))
    out = pd.Series("none", index=mtus, name="mtu_class")
    at = at[at.index.isin(mtus)]
    out.loc[at.index[at.to_numpy(bool)]] = "at_limit"
    out.loc[at.index[~at.to_numpy(bool)]] = "off_limit"
    return out


# ----------------------------------------------------------------------------------------------------- decode

def decode_duals(P: np.ndarray, p: np.ndarray, w_mu: float = 1e-3) -> tuple[float, np.ndarray]:
    """(lam, mu) from zonal prices: the LP of decode.fit_prices with one 'marginal unit' per zone.

    min sum_z |r_z| + w_mu * sum_c mu_c   s.t.  lam - sum_c mu_c P[c, z] + r_z = p_z,  mu >= 0.
    P: PTDF rows of the constraints (k x zones); p: zonal prices (or price differences: lam takes the level).
    fit_prices also bounds lam to [-1e4, 1e4] and each mu to [0, 1e4] EUR/MWh (decode.MAX_PRICE). The bound acts on
    the decoded values, not on the published ones: a set of rows that does not fix mu (for example a wrong active
    set) can push a decoded mu to the bound. analyse() counts these cases (summary key decoded_mu_at_bound).
    """
    from .decode import fit_prices

    n = P.shape[1]
    no = np.zeros(n, dtype=bool)
    return fit_prices(P, np.arange(n), p, ~no, no, no, w_mu=w_mu)


def identifiable(P: np.ndarray, tol: float = 1e-3) -> tuple[int, float]:
    """Numerical rank (singular values above `tol`) of the PTDF-difference matrix against DE, and the smallest
    singular value.

    If the rank is k, the price differences give at most one set of k duals. If it is smaller, more than one set
    can fit (mu >= 0 can still make the solution unique). A row of zeros (a constraint on a hub only) or two near
    copies of one row (one element under two contingencies) make the numerical rank smaller than k.
    """
    if P.shape[0] == 0:
        return 0, np.inf
    D = P - P[:, [list(ZONES.values()).index(REF)]]
    s = np.linalg.svd(D, compute_uv=False)
    return int((s > tol).sum()), float(s.min()) if len(s) >= P.shape[0] else 0.0


def zone_ptdf(df: pd.DataFrame) -> np.ndarray:
    return df[[f"ptdf_{z}" for z in ZONES.values()]].to_numpy(float).reshape(len(df), len(ZONES))


def compare(dec: np.ndarray, pub: np.ndarray) -> dict:
    """Decoded against published shadow prices (EUR/MWh). Relative errors only where the published price is above
    REL_MIN (n_rel rows)."""
    dec, pub = np.asarray(dec, float), np.asarray(pub, float)
    if len(pub) == 0:
        return {"n": 0}
    pos = pub > REL_MIN
    rel = np.abs(dec[pos] - pub[pos]) / pub[pos]
    return {"n": int(len(pub)), "corr": float(np.corrcoef(dec, pub)[0, 1]) if len(pub) > 2 and dec.std() > 0 else np.nan,
            "mae": float(np.abs(dec - pub).mean()), "median_abs_error": float(np.median(np.abs(dec - pub))),
            "mean_published": float(pub.mean()), "n_rel": int(pos.sum()),
            "median_rel_error": float(np.median(rel)) if pos.any() else np.nan,
            "share_within_10pct": float((rel <= 0.10).mean()) if pos.any() else np.nan}


# ----------------------------------------------------------------------------------------------------- (A) + (B)

def identity_and_decode(ac: pd.DataFrame, prices: pd.DataFrame, w_mu: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per MTU: (A) published duals -> price differences; (B) prices -> duals. Returns (per zone, per CNEC)."""
    zones = list(ZONES.values())
    i_ref = zones.index(REF)
    zrows, crows = [], []
    for t, g in ac.groupby("mtu"):
        if t not in prices.index or prices.loc[t].isna().any():
            continue
        p = prices.loc[t, zones].to_numpy(float)
        P, mu_pub = zone_ptdf(g), g.shadow_price.to_numpy(float)
        D = P - P[:, [i_ref]]
        pred = -(np.nan_to_num(mu_pub) @ D)  # (A): p_z - p_DE from the published duals (PL allocation rows: 0)
        lam, mu_dec = decode_duals(P, p, w_mu)  # (B)
        hub = g.hub_only.to_numpy(bool)
        rank_z, smin_z = identifiable(P[~hub])  # hub-only rows cannot be read from zonal prices: left out
        zrows.append(pd.DataFrame({"mtu": t, "zone": zones, "diff_actual": p - p[i_ref], "diff_identity": pred,
                                   "fit_error": lam - mu_dec @ P - p}))
        crows.append(pd.DataFrame({"mtu": t, "key": g.key.values, "tso": g.tso.values, "hub_only": hub,
                                   "published": mu_pub, "decoded": mu_dec, "k_zonal": int((~hub).sum()),
                                   "rank_zonal": rank_z, "smin_zonal": smin_z}))
    return pd.concat(zrows, ignore_index=True), pd.concat(crows, ignore_index=True)


def identity_errors(zones: pd.DataFrame) -> pd.DataFrame:
    """Per MTU: the largest |actual - identity| price-difference error over the non-PL zones and over all zones."""
    z = zones[zones.zone != REF]
    res = (z.diff_actual - z.diff_identity).abs()
    non_pl = (z.zone != "PL").to_numpy()
    return pd.DataFrame({"worst_non_pl": res[non_pl].groupby(z.mtu[non_pl]).max(),
                         "worst_all": res.groupby(z.mtu).max(),
                         "mean_non_pl": res[non_pl].groupby(z.mtu[non_pl]).mean()})


# ----------------------------------------------------------------------------------------------------- (C)

def match_published(fd: pd.DataFrame, g: pd.DataFrame, Pz: np.ndarray) -> np.ndarray:
    """Row of the final domain for each published CNEC (-1 if none): same element name, nearest (RAM, PTDF row).

    The contingency names differ between the two JAO tables, so they are not used. A match needs
    |RAM difference| / 2 MW + max |zonal PTDF difference| / 0.002 <= 1.
    """
    name = fd.cne_name.fillna("").str.strip().to_numpy()
    ram = fd.ram.to_numpy(float)
    out = np.full(len(g), -1)
    for i, (n_, r_, p_) in enumerate(zip(g.cnec_name.fillna("").str.strip(), g.ram.to_numpy(float), zone_ptdf(g))):
        cand = np.where(name == n_)[0]
        if len(cand) == 0:
            continue
        dist = np.abs(ram[cand] - r_) / 2.0 + np.abs(Pz[cand] - p_).max(axis=1) / 0.002
        j = int(np.argmin(dist))
        if dist[j] <= 1.0:
            out[i] = cand[j]
    return out


def primal(day: str, ac: pd.DataFrame, prices: pd.DataFrame, w_mu: float,
           alloc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(C) Read the active set from the primal: flow F = PTDF @ NP for every CNEC of the final domain.

    Returns (per MTU, scope and tolerance: precision, recall, decode errors; per published CNEC: its margin).
    The hourly domain applies to the four 15-min MTUs of the hour. Hubs enter the flow if they have both a
    PTDF column and a published net position. The flow omits ptdf_CH, because JAO publishes no CH net position.
    CNECs with the same RAM and PTDF row (one element under several contingencies) count as one constraint.
    A row is "binding" if RAM - flow <= tol; this includes violated rows (flow above RAM), which min_margin shows.
    min_margin of scope 'all' is the primal consistency test: no CNEC above its RAM by more than VIOLATION_MW.
    mae_union: mean |decoded - published| over the identified rows and the published rows that act on zonal
    prices (a published row that is not identified counts with decoded 0).
    `alloc`: PL allocation rows at their limit (net position = published limit, also a primal check). They are
    not in the final domain; the decode adds them to every identified set and leaves them out of the scores.
    """
    hours(day)  # refuses a day with a clock change
    npos = _read("net_position", day).rename(columns={"DE_LU": "DE"})
    zones = list(ZONES.values())
    rows, mrows = [], []
    for h in range(24):
        path = JAO_DIR / f"final_domain_{day}T{h:02d}.parquet"
        if not path.exists():
            continue
        fd = pd.read_parquet(path)
        nodes = [c[5:] for c in fd.columns if c.startswith("ptdf_") and c[5:] in npos.columns]
        H = fd[[f"ptdf_{z}" for z in nodes]].fillna(0.0).to_numpy(float)
        ram, pre = fd.ram.to_numpy(float), fd.presolved.to_numpy(bool)
        Pz = fd[[f"ptdf_{z}" for z in zones]].fillna(0.0).to_numpy(float)
        sig = pd.Series([hash(r.tobytes()) for r in np.c_[np.round(H, 4) + 0.0, np.round(ram, 1) + 0.0]])
        acts = np.abs(Pz - Pz[:, [zones.index(REF)]]).max(axis=1) > 1e-6  # changes zonal price differences
        # 'zonal': non-redundant, not one of the equality rows (ALEGrO pair, sum of hub positions), and with
        # some effect on zonal price differences (hub-only rows cannot be read from zonal prices)
        eq = fd.cne_name.fillna("").str.startswith("Equality Constraint").to_numpy()
        zonal = pre & ~eq & acts
        hour = pd.Timestamp(f"{day} {h:02d}:00", tz=TZ)
        for t in pd.date_range(hour, periods=4, freq="15min"):
            if t not in npos.index or t not in prices.index:
                continue
            margin = ram - H @ npos.loc[t, nodes].to_numpy(float)
            g = ac[(ac.mtu == t) & ac.shadow_price.notna()]
            m_idx = match_published(fd, g, Pz)
            for i, (k_, s_) in enumerate(zip(g.key, g.shadow_price)):
                mrows.append({"mtu": t, "key": k_, "published": s_, "found": m_idx[i] >= 0,
                              "presolved": bool(pre[m_idx[i]]) if m_idx[i] >= 0 else False,
                              "margin": float(margin[m_idx[i]]) if m_idx[i] >= 0 else np.nan})
            pub, pub_z = {}, {}  # published duals per signature; pub_z: only rows that act on zonal prices
            hub = g.hub_only.to_numpy(bool)
            for i in np.where(m_idx >= 0)[0]:
                for d_, use in [(pub, True), (pub_z, not hub[i])]:
                    if use:
                        d_[sig[m_idx[i]]] = d_.get(sig[m_idx[i]], 0.0) + float(g.shadow_price.iloc[i])
            n_miss = int((m_idx < 0).sum())  # unmatched published CNECs count as missed
            n_miss_z = int(((m_idx < 0) & ~hub).sum())
            miss_z = g.shadow_price.to_numpy(float)[(m_idx < 0) & ~hub]
            p = prices.loc[t, zones].to_numpy(float)
            Pa = zone_ptdf(alloc[alloc.mtu == t])
            for scope, cand in [("all", np.ones(len(fd), bool)), ("presolved", pre), ("zonal", zonal)]:
                ref = pub_z if scope == "zonal" else pub  # the zonal scope is scored on the zonal published rows
                n_pub = len(ref) + (n_miss_z if scope == "zonal" else n_miss)
                for tol in TOLS:
                    sel = np.where(cand & (margin <= tol))[0]
                    sel = sel[~sig.iloc[sel].duplicated().to_numpy()]
                    found = set(sig.iloc[sel])
                    hit = len(found & set(ref))
                    r = {"mtu": t, "scope": scope, "tol": tol, "n_found": len(sel), "n_published": n_pub,
                         "n_hit": hit, "min_margin": float(margin[cand].min())}
                    if len(sel) <= 200:  # decode with the identified set, compare on the union with published
                        lam, mu_all = decode_duals(np.vstack([Pz[sel], Pa]), p, w_mu)
                        mu = mu_all[:len(sel)]
                        dec = {k: v for k, v, a_ in zip(sig.iloc[sel], mu, acts[sel]) if a_}  # hub-only: not read
                        union = sorted(set(dec) | set(pub_z))
                        d_ = np.r_[[dec.get(k, 0.0) for k in union], np.zeros(len(miss_z))]
                        p_ = np.r_[[pub_z.get(k, 0.0) for k in union], miss_z]
                        r |= {"mae_union": float(np.abs(d_ - p_).mean()) if len(p_) else 0.0,
                              "fit_mae": float(np.abs(lam - mu_all @ np.vstack([Pz[sel], Pa]) - p).mean()),
                              "mu_max": float(mu_all.max()) if len(mu_all) else 0.0}
                    rows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(mrows)


def missed_extra(day: str, ac: pd.DataFrame, prices: pd.DataFrame, w_mu: float, alloc: pd.DataFrame) -> pd.DataFrame:
    """Effect of one missed or one extra constraint on the decoded duals (per MTU of `day`).

    missed: drop one published physical constraint: the one with the largest shadow price, the one with the
    smallest, and a random one (generator seeded with (SEED, MTU start in Unix seconds)). extra: add the
    non-redundant CNEC that is closest to its limit but not at it (RAM minus flow above 5 MW) and that changes
    zonal price differences. Dual errors are against the published shadow prices of the other constraints. The PL
    allocation rows at their limit (`alloc`) stay in every set; their duals are not scored.
    The price-fit errors follow from LP monotonicity (up to the small penalty on mu): a set with one row less
    cannot fit the prices better, and a set with one row more cannot fit them worse. Only the dual errors are
    evidence.
    """
    hours(day)  # refuses a day with a clock change
    npos = _read("net_position", day).rename(columns={"DE_LU": "DE"})
    zones = list(ZONES.values())
    rows = []
    for h in range(24):
        path = JAO_DIR / f"final_domain_{day}T{h:02d}.parquet"
        if not path.exists():
            continue
        fd = pd.read_parquet(path)
        nodes = [c[5:] for c in fd.columns if c.startswith("ptdf_") and c[5:] in npos.columns]
        H = fd[[f"ptdf_{z}" for z in nodes]].fillna(0.0).to_numpy(float)
        Pz = fd[[f"ptdf_{z}" for z in zones]].fillna(0.0).to_numpy(float)
        eq = fd.cne_name.fillna("").str.startswith("Equality Constraint").to_numpy()
        zonal = fd.presolved.to_numpy(bool) & ~eq & (np.abs(Pz - Pz[:, [zones.index(REF)]]).max(axis=1) > 1e-6)
        for t in pd.date_range(pd.Timestamp(f"{day} {h:02d}:00", tz=TZ), periods=4, freq="15min"):
            g = ac[(ac.mtu == t) & ~ac.hub_only]
            if len(g) < 2 or t not in prices.index or t not in npos.index:
                continue
            p = prices.loc[t, zones].to_numpy(float)
            P, pub = zone_ptdf(g), g.shadow_price.to_numpy(float)
            Pa = zone_ptdf(alloc[alloc.mtu == t])
            k = len(g)

            def dec(Pk: np.ndarray) -> tuple[np.ndarray, float]:
                """Duals of the rows of Pk (PL allocation rows added) and the mean price fit error."""
                lam_, mu_ = decode_duals(np.vstack([Pk, Pa]), p, w_mu)
                return mu_[:len(Pk)], float(np.abs(lam_ - mu_ @ np.vstack([Pk, Pa]) - p).mean())

            mu0, fit0 = dec(P)
            r = {"mtu": t, "k": k, "base_mae": float(np.abs(mu0 - pub).mean()), "base_fit_mae": fit0}
            rng = np.random.default_rng([SEED, int(t.timestamp())])
            for name, j in [("largest", int(np.argmax(pub))), ("smallest", int(np.argmin(pub))),
                            ("random", int(rng.integers(k)))]:
                keep = np.arange(k) != j
                mu1, fit1 = dec(P[keep])
                r |= {f"drop_{name}_published": float(pub[j]),
                      f"drop_{name}_mae_others": float(np.abs(mu1 - pub[keep]).mean()),
                      f"drop_{name}_base_mae_others": float(np.abs(mu0[keep] - pub[keep]).mean()),
                      f"drop_{name}_fit_mae": fit1}
            margin = fd.ram.to_numpy(float) - H @ npos.loc[t, nodes].to_numpy(float)
            published = set(ac.cnec_name[ac.mtu == t].fillna("").str.strip())
            cand = zonal & (margin > 5.0) & ~fd.cne_name.fillna("").str.strip().isin(published).to_numpy()
            if cand.any():
                e = int(np.where(cand)[0][np.argmin(margin[cand])])
                mu2, fit2 = dec(np.vstack([P, Pz[e]]))
                r |= {"extra_margin": float(margin[e]), "extra_decoded": float(mu2[-1]),
                      "extra_mae_others": float(np.abs(mu2[:-1] - pub).mean()), "extra_fit_mae": fit2}
            rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------- (D)

def predict_held_out(P: np.ndarray, p: np.ndarray, zone: int, w_mu: float) -> tuple[float, float]:
    """Fit (lam, mu) on every zone except `zone`; return the predicted price of `zone` and the largest mu."""
    keep = np.arange(P.shape[1]) != zone
    lam, mu = decode_duals(P[:, keep], p[keep], w_mu)
    return float(lam - mu @ P[:, zone]), float(mu.max()) if len(mu) else 0.0


def other_mtus(mtus: pd.Index, donors: list, seed: int = SEED) -> dict:
    """For each MTU, a random other MTU from `donors` (the MTUs with a published physical constraint), seed 0."""
    rng = np.random.default_rng(seed)
    pos = {t: i for i, t in enumerate(donors)}
    out = {}
    for t in mtus:
        if t in pos:
            i = int(rng.integers(len(donors) - 1))
            out[t] = donors[i + (i >= pos[t])]
        else:
            out[t] = donors[int(rng.integers(len(donors)))]
    return out


def leave_one_zone_out(ac: pd.DataFrame, alloc: pd.DataFrame, prices: pd.DataFrame, cls: pd.Series,
                       w_mu: float) -> pd.DataFrame:
    """(D) For each MTU and each zone except PL: fit lam and mu on the other zones, predict the held-out price.

    Two active sets: 'published' (the published physical constraints of the MTU) and 'other_mtu' (the published
    physical constraints of another MTU, with that MTU's PTDF rows; other_mtus, seed 0). Both sets get the PL
    allocation rows of the MTU itself, because these follow from the published net position. Hub-only rows are
    left out: they do not change zonal price differences. error = predicted - actual (EUR/MWh).
    """
    zones = list(ZONES.values())
    phys = ac[~ac.hub_only.astype(bool)]
    P_of = {t: zone_ptdf(g) for t, g in phys.groupby("mtu")}
    keys_of = {t: frozenset(g.key) for t, g in phys.groupby("mtu")}
    Pa_of = {t: zone_ptdf(g) for t, g in alloc.groupby("mtu")}
    empty = np.zeros((0, len(zones)))
    mtus = prices.index[prices[zones].notna().all(axis=1).to_numpy()]
    other = other_mtus(mtus, sorted(P_of))
    rows = []
    for t in mtus:
        p = prices.loc[t, zones].to_numpy(float)
        Pa = Pa_of.get(t, empty)
        o = other[t]
        same = keys_of.get(t, frozenset()) == keys_of[o]
        for name, P in [("published", P_of.get(t, empty)), ("other_mtu", P_of[o])]:
            A = np.vstack([P, Pa])
            for zi, z in enumerate(zones):
                if z == "PL":
                    continue
                pred, mu_max = predict_held_out(A, p, zi, w_mu)
                rows.append({"mtu": t, "day": t.strftime("%Y-%m-%d"), "mtu_class": cls.get(t, "none"),
                             "active_set": name, "other_mtu": o if name == "other_mtu" else pd.NaT,
                             "same_cnecs": same if name == "other_mtu" else True, "zone": z, "k": len(P),
                             "actual": p[zi], "predicted": pred, "error": pred - p[zi], "mu_max": mu_max})
    return pd.DataFrame(rows)


def day_bootstrap(err: pd.Series, day: pd.Series, mtu: pd.Series, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Mean, median and share within PRICE_TOL of |err|, with 95 % intervals from a bootstrap over whole days."""
    a, d = np.abs(np.asarray(err, float)), np.asarray(day)
    if len(a) == 0:
        return {"n": 0}
    days = np.unique(d)
    groups = [a[d == x] for x in days]
    rng = np.random.default_rng(seed)
    stats = np.empty((n_boot, 3))
    for b in range(n_boot):
        s = np.concatenate([groups[i] for i in rng.integers(len(days), size=len(days))])
        stats[b] = s.mean(), np.median(s), (s <= PRICE_TOL).mean()
    lo, hi = np.percentile(stats, [2.5, 97.5], axis=0)
    return {"n_zone_prices": int(len(a)), "mtus": int(pd.Series(mtu).nunique()), "days": int(len(days)),
            "mean_abs_error": float(a.mean()), "mean_abs_error_ci95": [float(lo[0]), float(hi[0])],
            "median_abs_error": float(np.median(a)), "median_abs_error_ci95": [float(lo[1]), float(hi[1])],
            "share_within_1": float((a <= PRICE_TOL).mean()), "share_within_1_ci95": [float(lo[2]), float(hi[2])],
            "n_above_1": int((a > PRICE_TOL).sum()), "max_abs_error": float(a.max())}


# ----------------------------------------------------------------------------------------------------- analyse

def class_facts(cls: pd.Series, ac: pd.DataFrame, pm: pd.DataFrame) -> dict:
    """Counts of the MTU classes, their hours, and the check of the margins against JAO's ram_mcp."""
    phys = pm[~pm.hub_only.astype(bool)]
    off = cls[cls == "off_limit"]
    day_hours = (off.index.hour >= 8) & (off.index.hour < 17)
    in_8_17 = (cls.index.hour >= 8) & (cls.index.hour < 17)
    with_c = cls[cls != "none"]
    g = phys.groupby("mtu").ram_mcp  # JAO's own margin after market coupling
    mcp = pd.DataFrame({"off": g.apply(lambda v: bool((v.abs() >= LIMIT_MW).any())),
                        "slack": g.max() >= LIMIT_MW, "over": g.min() <= -LIMIT_MW}).reindex(off.index)
    mcp = mcp.fillna(False).astype(bool)
    return {
        "mtus_with_prices": int(len(cls)),
        "mtus_with_published_physical_constraint": int(len(with_c)),
        "mtus_at_limit": int((cls == "at_limit").sum()), "mtus_off_limit": int(len(off)),
        "mtus_no_published_physical_constraint": int((cls == "none").sum()),
        "mtus_with_only_hub_only_published_rows": int(ac[ac.mtu.isin(cls.index[cls == "none"])].mtu.nunique()),
        "share_at_limit_of_mtus_with_constraint": float((with_c == "at_limit").mean()),
        "share_off_limit_of_mtus_with_constraint": float((with_c == "off_limit").mean()),
        "share_of_off_limit_mtus_starting_08_to_17": float(day_hours.mean()),
        "share_off_limit_of_mtus_with_constraint_08_to_17": float((with_c[in_8_17[cls != "none"]] == "off_limit").mean()),
        "off_limit_mtus_where_jao_ram_mcp_is_off_limit": int(mcp.off.sum()),
        "off_limit_mtus_with_ram_mcp_above_1mw_slack": int(mcp.slack.sum()),
        "off_limit_mtus_with_ram_mcp_below_minus_1mw_overload": int(mcp.over.sum()),
        "max_abs_margin_minus_ram_mcp_mw": float((phys.margin - phys.ram_mcp).abs().max()),
        "share_physical_rows_within_1mw": float((phys.margin.abs() < LIMIT_MW).mean()),
        "hub_only_rows_margin_over_ram_median": float((pm.margin / pm.ram)[pm.hub_only.astype(bool)].median()),
    }


def no_constraint_facts(prices: pd.DataFrame, cls: pd.Series, pl: pd.DataFrame, zones2: pd.DataFrame) -> dict:
    """MTUs with no published physical constraint: is every price split a PL split at a PL allocation limit?"""
    t = cls.index[(cls == "none").to_numpy()]
    p = prices.loc[t]
    non_pl = [z for z in prices.columns if z != "PL"]
    split = (p.max(axis=1) - p.min(axis=1)) > PRICE_TOL
    s = p[split]
    only_pl = (s[non_pl].max(axis=1) - s[non_pl].min(axis=1)) <= 0.01
    lim = pl.reindex(s.index).fillna(False).astype(bool)
    gap = s.PL - s[non_pl].mean(axis=1)  # PL price minus the common price of the other zones
    fit = zones2[zones2.mtu.isin(s.index) & (zones2.zone != REF)].fit_error.abs()
    return {
        "mtus": int(len(t)), "mtus_one_price_everywhere": int((~split).sum()),
        "mtus_price_split_above_1": int(split.sum()),
        "of_which_only_pl_differs": int(only_pl.sum()),
        "of_which_pl_at_export_limit": int(lim.at_export.sum()),
        "of_which_pl_at_import_limit": int(lim.at_import.sum()),
        "of_which_pl_price_below_the_others": int((gap < 0).sum()),
        "pl_minus_others_eur_mwh_min_max": [float(gap.min()), float(gap.max())] if len(gap) else [],
        "price_fit_mae_with_pl_allocation_row": float(fit.mean()) if len(fit) else np.nan,
    }


def analyse(days: list[str], domain_days: list[str], source: str, w_mu: float) -> None:
    from .decode import MAX_PRICE

    out = Path("results")
    skipped = [d for d in days if not (JAO_DIR / f"active_constraints_{d}.parquet").exists()]
    days = [d for d in days if d not in skipped]
    if source == "jao":
        prices = jao_prices(days)
    else:
        prices = ec_prices(days)
        prices = prices.sub(prices[REF], axis=0)
    try:  # cross-check of the two price sources: Energy-Charts zonal prices against JAO border spreads
        ec = ec_prices(days)
        ec = ec.sub(ec[REF], axis=0)
        jp = jao_prices(days).reindex(ec.index)
        cross = {"mtus": int(len(ec)), "max_abs_diff_eur_mwh": float((ec - jp).abs().max().max())}
    except (ValueError, KeyError) as e:  # Energy-Charts files missing
        cross = {"error": str(e)[:100]}
    ac = active(days)
    pl = pl_limits(days)
    alloc = allocation_rows(pl)
    pm = published_margins(ac, days)
    pm.to_csv(out / "market_core_published_margins.csv", index=False)
    cls = mtu_classes(pm, prices.index)
    plx = pl.reindex(prices.index).fillna(False).astype(bool)
    summary = {"days": days, "days_skipped": skipped, "price_source": source, "lp_penalty_w_mu": w_mu,
               "energy_charts_vs_jao_prices": cross,
               "class_rule": f"at_limit: every published physical constraint has |RAM - flow| < {LIMIT_MW} MW under "
                             "the published net positions; off_limit: at least one has not; none: no published "
                             "physical constraint",
               "mtu_classes": class_facts(cls, ac, pm),
               "pl_allocation_limits": {"mtus_pl_at_export_limit": int(plx.at_export.sum()),
                                        "mtus_pl_at_import_limit": int(plx.at_import.sum())}}

    zones, duals = identity_and_decode(ac, prices, w_mu)  # (A) + (B) with the published set
    zones2, duals2 = identity_and_decode(pd.concat([ac, alloc], ignore_index=True), prices, w_mu)  # + PL rows
    summary["mtus_without_published_constraint"] = no_constraint_facts(prices, cls, pl, zones2)

    ie = identity_errors(zones)  # (A)
    ie["mtu_class"] = cls.reindex(ie.index)
    ok = ie.worst_non_pl <= PRICE_TOL  # the published duals give the non-PL price differences
    ident = {}
    for name, sel in [("at_limit", ie.mtu_class == "at_limit"), ("off_limit", ie.mtu_class == "off_limit"),
                      ("all_with_constraint", ie.mtu_class != "none")]:
        ident[name] = {"mtus": int(sel.sum()), "mtus_non_pl_within_1": int(ok[sel].sum()),
                       "share_mtus_non_pl_within_1": float(ok[sel].mean()),
                       "share_mtus_all_zones_within_1": float((ie.worst_all[sel] <= PRICE_TOL).mean()),
                       "mean_abs_error_non_pl": float(ie.mean_non_pl[sel].mean())}
    z_ = zones[zones.zone != REF]
    ident["corr_all_zones"] = float(np.corrcoef(z_.diff_actual, z_.diff_identity)[0, 1])
    ident["mean_residual_by_zone"] = (z_.diff_actual - z_.diff_identity).groupby(z_.zone).mean().round(2).to_dict()
    summary["identity_A"] = ident

    # (B) with the PL allocation rows; in the identity-OK MTUs this is an identifiability check only
    d = duals2[duals2.published.notna() & ~duals2.hub_only.astype(bool)]
    okm = d.mtu.map(ok).fillna(False).astype(bool)
    mt = duals2[duals2.mtu.isin(set(duals.mtu))].groupby("mtu").first()  # MTUs with a published constraint
    summary["decode_B"] = {
        "note": "In the MTUs where (A) holds, (B) only shows that the duals are identifiable: not separate evidence",
        "price_fit_mae_published_set": float(zones[zones.zone != REF].fit_error.abs().mean()),
        "price_fit_mae_published_set_plus_pl_rows": float(zones2[zones2.zone != REF].fit_error.abs().mean()),
        "duals_identity_ok_mtus": compare(d.decoded[okm], d.published[okm]),
        "duals_other_mtus": compare(d.decoded[~okm], d.published[~okm]),
        "share_mtus_full_rank_physical_rows": float((mt.rank_zonal == mt.k_zonal).mean()),
        "hub_only_share_of_published_rows": float(duals.hub_only.astype(bool).mean()),
        "hub_only_share_of_published_shadow_price": float(duals.published[duals.hub_only.astype(bool)].sum()
                                                           / duals.published.sum())}
    zones.merge(zones2[["mtu", "zone", "fit_error"]].rename(columns={"fit_error": "fit_error_plus_pl"}),
                how="outer").assign(mtu_class=lambda x: x.mtu.map(cls)).to_csv(out / "market_core_zones.csv",
                                                                               index=False)
    duals2.rename(columns={"decoded": "decoded_plus_pl"}).merge(
        duals[["mtu", "key", "decoded"]], how="left").assign(identity_ok=lambda x: x.mtu.map(ok)).to_csv(
        out / "market_core_duals.csv", index=False)

    lz = leave_one_zone_out(ac, alloc, prices, cls, w_mu)  # (D)
    lz.to_csv(out / "market_core_leave_one_zone_out.csv", index=False)
    lo = {"rule": "each MTU and each zone except PL: fit lam and mu on the other zones, predict the held-out zone; "
                  "other_mtu = published physical rows of a random other MTU (seed 0); PL allocation rows of the MTU "
                  "itself in both sets; intervals: bootstrap over whole days, 2,000 draws, seed 0",
          "share_mtus_other_mtu_has_same_cnecs": float(lz[lz.active_set == "other_mtu"].groupby("mtu").same_cnecs
                                                       .first().mean())}
    for s in ["published", "other_mtu"]:
        g = lz[lz.active_set == s]
        lo[s] = {name: day_bootstrap(h.error, h.day, h.mtu) for name, h in
                 [("all_with_constraint", g[g.mtu_class != "none"]), ("at_limit", g[g.mtu_class == "at_limit"]),
                  ("off_limit", g[g.mtu_class == "off_limit"]), ("none", g[g.mtu_class == "none"]), ("all", g)]}
    summary["leave_one_zone_out_D"] = lo

    prim = pd.DataFrame()
    parts = []
    for day in domain_days:  # (C)
        if not list(JAO_DIR.glob(f"final_domain_{day}T*.parquet")):
            continue
        a_day = ac[ac.mtu.dt.strftime("%Y-%m-%d") == day]
        r, mg = primal(day, a_day, prices, w_mu, alloc)
        parts.append((r, mg, missed_extra(day, a_day, prices, w_mu, alloc)))
    if parts:
        prim, marg, me = (pd.concat(x, ignore_index=True) for x in zip(*parts))
        # consistent: no CNEC of the final domain above its RAM by more than VIOLATION_MW (primal only)
        cons = prim[prim.scope == "all"].groupby("mtu").min_margin.min() >= -VIOLATION_MW
        prim["primal_consistent"] = prim.mtu.map(cons).astype(bool)
        prim["mtu_class"] = prim.mtu.map(cls)
        me["mtu_class"] = me.mtu.map(cls)
        prim.to_csv(out / "market_core_primal.csv", index=False)
        marg.to_csv(out / "market_core_margins.csv", index=False)
        me.to_csv(out / "market_core_missed_extra.csv", index=False)
        sw = prim.groupby(["scope", "tol", "primal_consistent"]).agg(
            n_found=("n_found", "sum"), n_published=("n_published", "sum"), n_hit=("n_hit", "sum"),
            mae_union=("mae_union", "mean"), fit_mae=("fit_mae", "mean"), mtus=("mtu", "nunique")).reset_index()
        sw["precision"] = sw.n_hit / sw.n_found
        sw["recall"] = sw.n_hit / sw.n_published
        sw.to_csv(out / "market_core_primal_sweep.csv", index=False)
        head = prim[(prim.scope == "zonal") & (prim.tol == TOL_C)]

        def score(h: pd.DataFrame) -> dict:
            return {"mtus": int(h.mtu.nunique()), "recall": float(h.n_hit.sum() / h.n_published.sum()),
                    "precision": float(h.n_hit.sum() / h.n_found.sum()),
                    "mtus_exact_set": int(((h.n_hit == h.n_found) & (h.n_hit == h.n_published)).sum()),
                    "dual_mae_union_mean": float(h.mae_union.mean()),
                    "dual_mae_union_median": float(h.mae_union.median()),
                    "price_fit_mae_mean": float(h.fit_mae.mean())}

        found = marg[marg.found]
        ct = pd.crosstab(cons.rename("consistent"), cons.index.map(cls).rename("mtu_class"))
        summary["primal_C"] = {
            "domain_days": domain_days,
            "consistency_rule": f"no CNEC of the final domain has a flow above RAM + {VIOLATION_MW} MW under the "
                                "published net positions (primal only; does not use the published active set)",
            "headline_scope_and_tolerance": f"non-redundant CNECs that change zonal price differences; "
                                            f"RAM - flow <= {TOL_C} MW",
            "consistent": score(head[head.primal_consistent]), "other": score(head[~head.primal_consistent]),
            "pooled": score(head),
            "consistent_vs_mtu_class": {f"{'consistent' if a else 'inconsistent'}_{b}": int(ct.loc[a, b])
                                        for a in ct.index for b in ct.columns},
            "rules_chosen_on_scored_days": [
                f"tolerance {TOL_C} MW on RAM - flow (the sweep file shows 0.5 to 50 MW)",
                "scope: non-redundant (presolved) CNECs, no equality rows, with an effect on zonal price differences",
                f"consistency threshold {VIOLATION_MW} MW",
                "match of a published CNEC to the final domain: |RAM difference| / 2 MW + max |PTDF difference| / "
                "0.002 <= 1"],
            "published_found_in_domain": float(marg.found.mean()),
            "published_margin_quantiles_mw": dict(zip(["min", "p05", "median", "p95", "max"],
                                                      np.quantile(found.margin, [0, .05, .5, .95, 1]).round(2).tolist())),
            "sweep_csv": "results/market_core_primal_sweep.csv"}
        cols = [c for c in me.columns if c not in ("mtu", "mtu_class")]
        summary["missed_extra"] = {
            "note": "price-fit errors follow from LP monotonicity (one row less cannot fit better, one row more cannot "
                    "fit worse); only the dual errors (mae_others) are evidence. Classes as in mtu_classes.",
            **{c: {"mtus": int((me.mtu_class == c).sum()),
                   "median": me[me.mtu_class == c][cols].median().round(3).to_dict(),
                   "mean": me[me.mtu_class == c][cols].mean().round(3).to_dict()} for c in ["at_limit", "off_limit"]}}

    def n_bound(v: pd.Series) -> int:
        return int((np.asarray(v, float) >= 0.999 * MAX_PRICE).sum())

    summary["decoded_mu_at_bound"] = {
        "bound_eur_mwh": MAX_PRICE, "largest_published_shadow_price": float(ac.shadow_price.max()),
        "decode_B_rows": n_bound(duals2.decoded),
        "leave_one_zone_out_published_fits": n_bound(lz.mu_max[lz.active_set == "published"]),
        "leave_one_zone_out_other_mtu_fits": n_bound(lz.mu_max[lz.active_set == "other_mtu"]),
        "primal_C_fits": n_bound(prim.mu_max.dropna()) if len(prim) else 0}
    (out / "market_core_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({k: summary[k] for k in ["mtu_classes", "identity_A", "decoded_mu_at_bound"]}, indent=1,
                     default=float))
    figure(duals2, ok, lz)


def figure(duals: pd.DataFrame, identity_ok: pd.Series, lz: pd.DataFrame,
           path: Path = Path("figures/fig3_core_duals.png")) -> None:
    """Figure 3: (a) decoded against published shadow prices; (b) leave-one-zone-out error, right and wrong set."""
    import matplotlib.pyplot as plt

    INK, SURF, BLUE, ORANGE, GREY = "#0b0b0b", "#fcfcfb", "#2a78d6", "#eb6834", "#8a8984"
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.edgecolor": GREY,
                         "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                         "axes.facecolor": SURF, "savefig.dpi": 200, "savefig.facecolor": SURF})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5.3))
    floor = 0.1
    d = duals[duals.published.notna() & ~duals.hub_only.astype(bool) & (duals.published > floor)]
    ok = d.mtu.map(identity_ok).fillna(False).astype(bool).to_numpy()
    ax.plot([floor, 5e3], [floor, 5e3], color=GREY, lw=0.8, zorder=1)
    for sel, color, label in [(~ok, ORANGE, "published duals miss the prices"),
                              (ok, BLUE, "published duals give the prices")]:
        ax.scatter(d.published[sel], np.maximum(d.decoded[sel], floor), s=7, alpha=0.45, color=color,
                   edgecolors="none", zorder=2, label=f"{label} (n = {sel.sum():,} pairs in "
                                                      f"{d.mtu[sel].nunique():,} quarter-hours)")
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlim(floor, 5e3), ax.set_ylim(floor * 0.8, 5e3)
    ax.set_xlabel("Published shadow price (EUR/MWh)")
    ax.set_ylabel("Shadow price decoded from zonal prices (EUR/MWh)")
    ax.set_title("(a) Decoded against published shadow prices", loc="left", fontsize=10)
    ax.text(4e3, 0.13, "decoded values below 0.1\nare drawn at 0.1", color=GREY, fontsize=8.5, ha="right")
    leg = ax.legend(frameon=False, fontsize=8.5, markerscale=2.5, loc="upper left", bbox_to_anchor=(0.0, -0.13),
                    title="One point per constraint and quarter-hour, decoded with the published active set.\n"
                          "Blue: the published duals give the price differences (Poland aside) within 1 EUR/MWh.", title_fontsize=8.5,
                    alignment="left")
    for h in leg.legend_handles:
        h.set_alpha(1.0)

    floor_b = 1e-4
    for s, color, slabel in [("published", BLUE, "published active set"),
                             ("other_mtu", ORANGE, "active set of another quarter-hour")]:
        for c, ls, clabel in [("at_limit", "-", "all at their limit"),
                              ("off_limit", "--", "some off their limit")]:
            e = np.sort(np.maximum(lz.error[(lz.active_set == s) & (lz.mtu_class == c)].abs().to_numpy(), floor_b))
            bx.step(e, np.arange(1, len(e) + 1) / len(e), where="post", color=color, ls=ls, lw=1.8,
                    label=f"{slabel}, {clabel} (n = {len(e):,})")
    bx.axvline(PRICE_TOL, color=GREY, lw=0.8, ls=":", zorder=0)
    bx.text(PRICE_TOL * 0.85, 0.5, "1 EUR/MWh", color=GREY, fontsize=8.5, ha="right")
    x_max = max(lz.error.abs().max() * 1.5, 10.0)
    bx.text(x_max * 0.8, 0.04, "errors below 0.0001\nare drawn at 0.0001", color=GREY, fontsize=8.5, ha="right")
    bx.set_xscale("log")
    bx.set_xlim(floor_b * 0.7, x_max)
    bx.set_ylim(0, 1.02)
    bx.set_xlabel("Error of the held-out zone price (EUR/MWh)")
    bx.set_ylabel("Share with a smaller error")
    bx.set_title("(b) Leave one zone out: price of the held-out zone", loc="left", fontsize=10)
    bx.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.0, -0.13),
              title="n = held-out zone prices (11 zones per quarter-hour; Poland not held out)", title_fontsize=8.5,
              alignment="left")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.31, wspace=0.22)
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path)
    print(f"saved {path}")


def figure_from_results(out: Path = Path("results")) -> None:
    """Figure 3 from the files that analyse() writes (no JAO data needed)."""
    duals = pd.read_csv(out / "market_core_duals.csv").rename(columns={"decoded_plus_pl": "decoded_b"})
    duals = duals.drop(columns="decoded").rename(columns={"decoded_b": "decoded"})
    identity_ok = duals.dropna(subset=["identity_ok"]).groupby("mtu").identity_ok.first().astype(bool)
    figure(duals, identity_ok, pd.read_csv(out / "market_core_leave_one_zone_out.csv"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["fetch", "analyse", "figure"], help="figure: Figure 3 from results/ only")
    ap.add_argument("--days", nargs="+", default=DAYS, help="business days YYYY-MM-DD")
    ap.add_argument("--domain-days", nargs="+", default=DOMAIN_DAYS, help="days with the final domain, for (C)")
    ap.add_argument("--prices", choices=["jao", "ec"], default="jao",
                    help="jao: JAO border price spreads (default); ec: Energy-Charts zonal prices")
    ap.add_argument("--w-mu", type=float, default=1e-3, help="penalty on the sum of the duals in the LP")
    a = ap.parse_args()
    if a.step == "fetch":
        failed = fetch_jao(a.days)
        for d in a.domain_days:
            failed += fetch_domain(d)
        fetch_prices(a.days[0], a.days[-1])
        print("failed (skipped):", failed)
    elif a.step == "figure":
        figure_from_results()
    else:
        analyse(a.days, a.domain_days, a.prices, a.w_mu)


if __name__ == "__main__":
    main()
