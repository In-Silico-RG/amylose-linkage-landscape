#!/usr/bin/env python3
"""Compare the DFT benchmark with the xtb map and build a DFT-corrected map (2026-09-23).
Correction model: dE(phi, psi) = a + b*psi_H on syn cells (fitted on the valley cells) and a constant
shift on anti cells (fitted on the band-flip minima); applied to every valid cell of the ALPB map;
exact RIS recomputed. Output ../resultados/dft_compare.json and a printed table."""
import json, sys, os
import numpy as np
SRC = sys.argv[1] if len(sys.argv) > 1 else "../resultados/dft_benchmark_r2scan_def2-svp.json"   # optional: another benchmark file (e.g. the TZVP one)
TAG = "" if "def2-svp" in SRC else "_" + os.path.basename(SRC).replace("dft_benchmark_", "").replace(".json", "")
sys.argv = ["x"]
import mc_chains as mc
from chain_builder import Template
H2K = 627.509; kT = 0.0019872 * 303.15
d = json.load(open(SRC)); print("source:", SRC)
cells = {k: (v["phi"], v["psi"], H2K * v["E_dft"], H2K * v["E_xtb"]) for k, v in d.items() if v["converged"]}
ref = cells["100,-150"]
rows = {k: (p, s, ed - ref[2], ex - ref[3]) for k, (p, s, ed, ex) in cells.items()}
psiH = lambda s: ((s + 117.8 + 180) % 360) - 180
regions = {"valley": [k for k in rows if rows[k][0] == 100 and rows[k][1] <= -110 and rows[k][1] >= -170],
           "phi wall": [k for k in rows if rows[k][1] == -150 and rows[k][0] != 100],
           "band flip": ["90,70", "60,50"], "barrier X": ["100,120", "100,130", "100,140"], "barrier Y": ["100,-30", "100,-20", "100,-10"],
           "secondary": ["160,-110", "60,-170", "170,-60"]}
print(f"{'region':10s} {'cell':10s} {'DFT':>7s} {'xtb':>7s} {'DFT-xtb':>8s}  (kcal/mol, relative to (100,-150))")
diff = {}
for reg, ks in regions.items():
    for k in ks:
        if k in rows:
            p, s, ed, ex = rows[k]; diff[k] = ed - ex; print(f"{reg:10s} {k:10s} {ed:7.2f} {ex:7.2f} {ed-ex:8.2f}")
# valley tilt fit: DFT-xtb vs psi_H on the valley cells
vk = [k for k in regions["valley"] if k in rows]
x = np.array([psiH(rows[k][1]) for k in vk]); y = np.array([diff[k] for k in vk])
b, a = np.polyfit(x, y, 1); print(f"\nvalley correction: dE = {a:.2f} + {b:.3f} * psi_H  (kcal/mol; slope {b*10:.2f} per 10 deg); rms residual {np.std(y-(a+b*x)):.2f}")
anti_shift = diff["90,70"] if "90,70" in diff else 0.0; print(f"band-flip correction (cell (90,70)): {anti_shift:+.2f} kcal/mol" + ("" if "90,70" in diff else "  [cell not in this set: no anti shift applied]"))
# apply to the map
T = Template(str(mc.HERE / "maltose_a_84_-145.xyz")); rng = np.random.default_rng(1)
pp, w, E = mc.load_map("scan2_10deg.dat"); M = mc.precompute(T, pp, rng); M[:, :, :3, 3] /= 10
pH = psiH(pp[:, 1]); syn = np.abs(pH) <= 90
lo, hi = x.min(), x.max()   # sampled psi_H range of the valley cells; hold the correction constant outside it
E2 = E + np.where(syn, a + b * np.clip(pH, lo, hi), anti_shift)
print(f"correction applied with psi_H clamped to the sampled range [{lo:.0f}, {hi:.0f}] deg")
def stats(EE):
    ww = np.exp(-EE / kT); ww /= ww.sum(); r = mc.ris_exact(M, ww, T, nlist=(100,))
    ws = ww * syn; mpsi = np.degrees(np.arctan2((ws * np.sin(np.radians(pH))).sum(), (ws * np.cos(np.radians(pH))).sum()))
    return dict(Cinf=r["Cinf"], period=r["period_bonds"], lam=max(r["lam"]), c1=r["corr"][1], c2=r["corr"][2], p_anti=float(ww[~syn].sum()), mean_psiH=float(mpsi))
base = stats(E); corr = stats(E2)
print("\nmap as computed :", {k: round(v, 2) for k, v in base.items()})
print("DFT-corrected   :", {k: round(v, 2) for k, v in corr.items()})
# LaTeX table for the SI (energies relative to the minimum of each method over the sampled cells)
emin_d = min(r[2] for r in rows.values()); emin_x = min(r[3] for r in rows.values())
names = {"valley": "syn valley, $\\varphi=100^\\circ$", "phi wall": "$\\varphi$ wall, $\\psi=-150^\\circ$", "band flip": "band-flip minima", "barrier X": "barrier X ($\\psi_H \\approx -120^\\circ$)", "barrier Y": "barrier Y ($\\psi_H \\approx +100^\\circ$)", "secondary": "secondary minima"}
L = []
for reg, ks in regions.items():
    first = True
    for k in ks:
        if k not in rows: continue
        p_, s_, ed, ex = rows[k]; t = d[k]["seconds"] / 60
        L.append(f"{names[reg] if first else ''} & $({p_},{s_})$ & ${psiH(s_):.0f}$ & {ed-emin_d:.1f} & {ex-emin_x:.1f} & {ed-ex-(emin_d-emin_x):+.1f} & {t:.0f} \\\\")
        first = False
open(f"../manuscript_JPCB/tab_dft_benchmark{TAG}.tex", "w").write("\n".join(L) + "\n\\bottomrule\n")
print(f"table written: {len(L)} rows; DFT min at", min(rows, key=lambda k: rows[k][2]), "xtb min at", min(rows, key=lambda k: rows[k][3]))
json.dump(dict(rows=rows, diff=diff, valley_fit=dict(a=a, b=b), anti_shift=anti_shift, base=base, corrected=corr), open(f"../resultados/dft_compare{TAG}.json", "w"), indent=1)
