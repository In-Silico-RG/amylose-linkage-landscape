#!/usr/bin/env python3
"""Ring-flexibility factor per rigid template on the ten 100-ns CHARMM36 replicas (2026-09-30).
Factor = C_inf(flexible residue frames) / C_inf(rigid template, same torsions), every 50 ps after 2 ns,
per replica; mean +- SE over the replicas; flexibility-corrected map value = map_rigid(template) x factor.
map_rigid is taken from ring_factor_templates.json (2026-09-25, the 10-ns version of this test).
Needs the chem env (RDKit chain builder) and ../resultados/rep/<tag>.npz from replica_analysis.py extract.
    -> ../resultados/ring_factor_templates_rep.json
"""
import json, sys
import numpy as np
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent; RES = HERE.parent / "resultados"
sys.path.insert(0, str(HERE))
TAGS = [f"SC{n}_r{r}" for n in (6, 12) for r in (2, 4, 6, 8, 10)]
STRIDE, T0 = 5, 2000.0


def one(a):
    tpl, tag = a
    import ring_factor_blocks as rfb
    rfb.TEMPLATE = tpl
    d = np.load(RES / "rep" / f"{tag}.npz"); m = d["t"] >= T0 - 1e-6
    R, v, pp = d["R"][m][::STRIDE].astype(float), d["v"][m][::STRIDE].astype(float), d["phipsi"][m][::STRIDE].astype(float)
    Rr, Vr = rfb.rigid_RV(pp.reshape(-1, 2))
    cf = rfb.cinf(R.mean((0, 1)), v.mean((0, 1)), np.mean(np.sum(v ** 2, -1)))
    cr = rfb.cinf(Rr.mean(0), Vr.mean(0), np.mean(np.sum(Vr ** 2, -1)))
    return tpl, tag, cf, cr


if __name__ == "__main__":
    old = json.load(open(RES / "ring_factor_templates.json"))
    with Pool(8) as p: rows = p.map(one, [(t, g) for t in old for g in TAGS])
    out = {}
    for t in old:
        f = np.array([cf / cr for tt, g, cf, cr in rows if tt == t]); cr = np.array([c for tt, g, _, c in rows if tt == t])
        out[t] = dict(map_rigid=old[t]["map_rigid"], factor=float(f.mean()), factor_se=float(f.std(ddof=1) / np.sqrt(len(f))),
                      md_rigid=float(cr.mean()), map_flex=float(old[t]["map_rigid"] * f.mean()), per_replica=f.tolist())
        print(f"{t:24s} map rigid {old[t]['map_rigid']:.2f}  factor {f.mean():.3f} +- {f.std(ddof=1)/np.sqrt(len(f)):.3f}  map x factor {out[t]['map_flex']:.2f}  (10-ns value {old[t]['factor']:.3f}, {old[t]['map_flex']:.2f})")
    json.dump(out, open(RES / "ring_factor_templates_rep.json", "w"), indent=1)
