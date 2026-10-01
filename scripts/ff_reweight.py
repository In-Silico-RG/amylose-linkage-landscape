#!/usr/bin/env python3
"""Do the explicit torsion terms account for the differences between CHARMM36 and GLYCAM06? (AFC 2026-10-01:
"the comparison between FFs should account for the differences")
Perturbation test: the (phi, psi) samples of one force field are reweighted by exp(-dU/kT), where dU is the sum of
the OTHER force field's explicit torsion terms about the two glycosidic bonds minus its own (ff_torsions.py). If
the torsion terms were what separates the force fields, the reweighted statistics would move onto the other
force field's. Everything else (non-bonded terms, water, ring terms) is left as sampled.
    -> ../resultados/ff_reweight.json     (chem env)"""
import glob, json, sys
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; RES = HERE.parent / "resultados"
sys.path.insert(0, str(HERE))
from ring_factor_blocks import rigid_RV, cinf
from ff_torsions import total
kT = 0.0019872 * 303.15
T = json.loads((RES / "ff_torsions.json").read_text()); tt, off = T["terms"], T["offsets"]
U = lambda ff, P: total(tt[ff]["phi"], off["phi"], P[:, 0]) + total(tt[ff]["psi"], off["psi"], P[:, 1])
load = lambda pat: np.concatenate([np.load(f)["phipsi"][200::10].reshape(-1, 2).astype(float) for f in sorted(glob.glob(str(RES / "rep" / pat)))])
sets = {"CHARMM36": load("SC*_r[0-9]*.npz"), "GLYCAM06": load("GLY6_r*.npz")}
def stats(P, R, V, w):
    w = w / w.sum(); syn = np.abs((P[:, 1] + 117.8 + 180) % 360 - 180) <= 90; ws = w[syn] / w[syn].sum()
    def cm(a):
        r = np.radians(a); m = np.degrees(np.arctan2((ws * np.sin(r)).sum(), (ws * np.cos(r)).sum())); d = (a - m + 180) % 360 - 180
        return float(m), float(np.sqrt((ws * d ** 2).sum()))
    (mphi, sphi), (mpsi, spsi) = cm(P[syn, 0]), cm(P[syn, 1])
    Rm = np.einsum("i,iab->ab", w, R); vm = w @ V; l2 = float(w @ np.sum(V ** 2, 1))
    return dict(flips=float(w[~syn].sum()), phi_mean=mphi, phi_sd=sphi, psi_mean=mpsi, psi_sd=spsi, Cinf=cinf(Rm, vm, l2), n_eff=float(1 / np.sum(w ** 2)))
out = {}
for k, other in (("CHARMM36", "GLYCAM06"), ("GLYCAM06", "CHARMM36")):
    P = sets[k]; R, V = rigid_RV(P); dU = U(other, P) - U(k, P)
    out[k] = dict(as_sampled=stats(P, R, V, np.ones(len(P))), with_torsion_terms_of_other=stats(P, R, V, np.exp(-(dU - dU.min()) / kT)),
                  dU_mean=float(dU.mean()), dU_sd=float(dU.std()))
    for a in ("as_sampled", "with_torsion_terms_of_other"): print(k, a, {q: round(v, 2) for q, v in out[k][a].items()}, flush=True)
(RES / "ff_reweight.json").write_text(json.dumps(out, indent=1))
