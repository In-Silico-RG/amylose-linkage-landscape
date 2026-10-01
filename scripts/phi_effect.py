#!/usr/bin/env python3
"""Does the mean phi matter for C_inf? (AFC 2026-10-01: "phi remains unchanged?")
Rigid-template C_inf (paper's template, exact residue-frame formula) from the torsions sampled in the long runs,
every 10th frame after 2 ns: all joints; unflipped joints only; and with phi of every sample shifted so that the
two force fields exchange their mean phi. chem env.  -> ../resultados/phi_effect.json"""
import glob, json, sys
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; RES = HERE.parent / "resultados"
sys.path.insert(0, str(HERE))
from ring_factor_blocks import rigid_RV, cinf
def load(pat):
    return np.concatenate([np.load(f)["phipsi"][200::10].reshape(-1, 2).astype(float) for f in sorted(glob.glob(str(RES / "rep" / pat)))])
def C(pp):
    R, V = rigid_RV(pp); return cinf(R.mean(0), V.mean(0), np.mean(np.sum(V ** 2, 1)))
def cmean(a):
    r = np.radians(a); return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())))
out = {}
sets = {"CHARMM36": load("SC*_r[0-9]*.npz"), "GLYCAM06": load("GLY6_r*.npz")}
syn = {k: np.abs((v[:, 1] + 117.8 + 180) % 360 - 180) <= 90 for k, v in sets.items()}
mphi = {k: cmean(v[syn[k], 0]) for k, v in sets.items()}; mpsi = {k: cmean(v[syn[k], 1]) for k, v in sets.items()}
for k, other in (("CHARMM36", "GLYCAM06"), ("GLYCAM06", "CHARMM36")):
    P = sets[k]; S = P[syn[k]]
    sh = S.copy(); sh[:, 0] += mphi[other] - mphi[k]
    sh2 = S.copy(); sh2[:, 1] += mpsi[other] - mpsi[k]
    out[k] = dict(n=int(len(P)), flips=float(1 - syn[k].mean()), phi_mean=mphi[k], psi_mean=mpsi[k],
                  phi_sd=float(np.std((S[:, 0] - mphi[k] + 180) % 360 - 180)), psi_sd=float(np.std((S[:, 1] - mpsi[k] + 180) % 360 - 180)),
                  C_all=C(P), C_unflipped=C(S), C_unflipped_phi_of_other=C(sh), C_unflipped_psi_of_other=C(sh2))
    print(k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in out[k].items()}, flush=True)
(RES / "phi_effect.json").write_text(json.dumps(out, indent=1))

# width test: unflipped CHARMM36 samples with their deviations from the mean rescaled to GLYCAM06's spreads (and the reverse)
wrap = lambda a: (a + 180) % 360 - 180
for k, other in (("CHARMM36", "GLYCAM06"), ("GLYCAM06", "CHARMM36")):
    S = sets[k][syn[k]].copy()
    for j, (m, sd, sdo) in enumerate(((mphi[k], out[k]["phi_sd"], out[other]["phi_sd"]), (mpsi[k], out[k]["psi_sd"], out[other]["psi_sd"]))):
        S[:, j] = m + wrap(S[:, j] - m) * sdo / sd
    out[k]["C_unflipped_widths_of_other"] = C(S)
    S[:, 0] += mphi[other] - mphi[k]; S[:, 1] += mpsi[other] - mpsi[k]; out[k]["C_unflipped_widths_and_means_of_other"] = C(S)
    print(k, "widths of", other, round(out[k]["C_unflipped_widths_of_other"], 2), "| widths and means:", round(out[k]["C_unflipped_widths_and_means_of_other"], 2), flush=True)
(RES / "phi_effect.json").write_text(json.dumps(out, indent=1))
