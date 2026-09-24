#!/usr/bin/env python3
"""Autocorrelation times of the LOCAL descriptors (referee A8): per-frame chain means of c1 (bond i . bond i+1),
of cos(psi) and of the band-flip count, over the analysed window (t >= 2 ns, every 20 ps) and over the full
10 ps trajectory. Output ../resultados/local_tau.json"""
import sys, json
import numpy as np, MDAnalysis as mda
from chain_stats import SYSTEMS, strands, make_chain_whole
from pathlib import Path
HERE = Path(__file__).resolve().parent


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2; b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n; w = b2 - np.dot(b2, b1n) * b1n
    return np.degrees(np.arctan2(np.dot(np.cross(b1n, v), w), np.dot(v, w)))


def tau(x, dt):
    x = x - x.mean(); n = len(x); v = x @ x / n
    if v == 0: return 0.0
    ac = [1.0]
    for k in range(1, n // 2):
        c = (x[:-k] @ x[k:]) / (n - k) / v
        if c <= 0: break
        ac.append(c)
    return dt * (1 + 2 * sum(ac[1:]))


out = {}
for s in sys.argv[1:]:
    top, trj = SYSTEMS[s]; u = mda.Universe(str(top), str(trj)); chains = strands(u)
    c1, cps, nfl, T = [], [], [], []
    for ts in u.trajectory:
        box = ts.dimensions[:3]; a = []; b = []; f = 0
        for c in chains:
            raw = np.array([r.atoms.select_atoms("name O4").positions[0] for r in c]); O4 = make_chain_whole(raw, box)
            bv = np.diff(O4, axis=0); bh = bv / np.linalg.norm(bv, axis=1)[:, None]
            a.append(np.mean(np.sum(bh[:-1] * bh[1:], axis=1)))
            for i in range(len(c) - 1):
                P = {nm: c[i + 1].atoms.select_atoms("name " + nm).positions[0] for nm in ("C1",)}
                Q = {nm: c[i].atoms.select_atoms("name " + nm).positions[0] for nm in ("O4", "C4", "C5")}
                for d in (P, Q):
                    for nm in d:
                        dd = d[nm] - raw[i]; dd -= box * np.round(dd / box); d[nm] = raw[i] + dd
                psi = dihedral(P["C1"], Q["O4"], Q["C4"], Q["C5"]); b.append(np.cos(np.radians(psi)))
                f += abs(((psi + 117.8 + 180) % 360 - 180)) > 90
        c1.append(np.mean(a)); cps.append(np.mean(b)); nfl.append(f); T.append(ts.time)
    T = np.array(T); dt = T[1] - T[0]; w = T >= 2000
    res = {}
    for name, x in (("c1", np.array(c1)), ("cos_psi", np.array(cps)), ("band_flips", np.array(nfl, float))):
        res[name] = dict(tau_full_ps=tau(x, dt), tau_window_ps=tau(x[w], dt), mean_window=float(x[w].mean()), sd_window=float(x[w].std()))
    out[s] = dict(dt_ps=float(dt), frames=len(T), **res)
    print(s, {k: (round(v["tau_full_ps"]), round(v["tau_window_ps"])) for k, v in res.items()}, flush=True)
json.dump(out, open(HERE.parent / "resultados" / "local_tau.json", "w"), indent=1)
