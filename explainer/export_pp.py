#!/usr/bin/env python3
"""(phi, psi) pairs visited in the long simulations, for the explainer: 2500 random linkage samples per force field
(after 2 ns), and the correlation between phi and psi among the unflipped linkages.  -> pp.json (merged by build.py)"""
import glob, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; REP = HERE.parent / "resultados" / "rep"
rng = np.random.default_rng(7); out = {}
for key, pat in (("CHARMM36", "SC*_r[0-9]*.npz"), ("GLYCAM06", "GLY6_r*.npz")):
    P = np.concatenate([np.load(f)["phipsi"][200:].reshape(-1, 2) for f in sorted(glob.glob(str(REP / pat)))]).astype(float)
    syn = np.abs((P[:, 1] + 117.8 + 180) % 360 - 180) <= 90; S = P[syn]
    psi = np.where(S[:, 1] > 100, S[:, 1] - 360, S[:, 1])                       # unwrap across +-180 before correlating
    r = float(np.corrcoef(S[:, 0], psi)[0, 1]); slope = float(np.polyfit(S[:, 0], psi, 1)[0])
    pick = P[rng.choice(len(P), 2500, replace=False)]
    out[key] = dict(pts=[[int(round(a)), int(round(b))] for a, b in pick], r=round(r, 2), slope=round(slope, 2), n=int(len(P)))
    print(key, "n", len(P), "corr(phi, psi) unflipped", round(r, 2), "slope dpsi/dphi", round(slope, 2))
(HERE / "pp.json").write_text(json.dumps(out, separators=(",", ":"))); print("bytes", (HERE / "pp.json").stat().st_size)
