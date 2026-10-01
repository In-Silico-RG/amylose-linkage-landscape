#!/usr/bin/env python3
"""Syn-valley and band-flip energies after hydroxyl-network sampling, and their consequence for the map
C_inf (2026-09-30).
Inputs: ../resultados/oh_sampling.json and oh_sampling_anti.json (oh_sampling.py final: lowest of 6 networks
per cell, r2SCAN-3c constrained opt and r2SCAN-D3(BJ)/def2-TZVP on it), oh_sampling{,_anti}/xtb_summary.json
and the optimised frames (xtb/ALPB: best of ~300 networks), ../resultados/dft_orca_r2scan.json (DFT r3: one
network per cell, the xtb map frame's), ../resultados/dft_benchmark_r2scan_def2-svp.json (PySCF set, xtb E).
Correction model as dft_compare.py but piecewise linear: dE(psi_H) = E_level(cell) - E_xtb-map-frame(cell)
on the seven valley cells (phi = 100), interpolated in psi_H, held constant outside the sampled range,
applied to every syn cell of the ALPB map (phi-independent). Anti cells: constant shift, that of the cell
(90,70) for phi > 75 and that of (60,50) for phi <= 75 ('anti shifted'); also 'p fixed' (anti weights
rescaled to the map's band-flip population: second number held, first moved).
Output ../resultados/oh_compare.json, three LaTeX tables in ../manuscript_JPCB/ and a printed table."""
import json, sys, re
import numpy as np
sys.argv = ["x"]
import mc_chains as mc
from chain_builder import Template
H2K = 627.509; kT = 0.0019872 * 303.15
R = mc.HERE.parent / "resultados"; TEX = mc.HERE.parent / "manuscript_JPCB"
r3 = json.load(open(R / "dft_orca_r2scan.json")); sv = json.load(open(R / "dft_benchmark_r2scan_def2-svp.json"))
oh = json.load(open(R / "oh_sampling.json"))
VAL = [f"100,{p}" for p in range(-170, -100, 10)]; ANTI = ["90,70", "60,50"]
have_anti = (R / "oh_sampling_anti.json").exists()
if have_anti:
    oha = json.load(open(R / "oh_sampling_anti.json")); have_anti = all("4_sp_opt" in oha.get(c, {}) and len(oha[c]["4_sp_opt"]["all"]) == 6 for c in ANTI)
    if have_anti: oh.update(oha)
cells = VAL + ANTI
psiH = lambda s: ((s + 117.8 + 180) % 360) - 180
x = np.array([psiH(int(c.split(",")[1])) for c in VAL])


def xtb_best(d, c):
    """Absolute xtb/ALPB energy (kcal/mol) of the lowest valid network of a cell and its distance to the map frame's."""
    s = json.load(open(mc.HERE / d / "xtb_summary.json"))[c]; f = mc.HERE / d / f"c_{c.replace(',', '_')}" / f"{s['kept'][0]['start']}.opt.xyz"
    return H2K * float(re.search(r"energy:\s+(\S+)", f.read_text().split("\n")[1]).group(1)), s["xtb_frame"], s["n_valid"]


xb = {c: xtb_best("oh_sampling", c) for c in VAL}
if (mc.HERE / "oh_sampling_anti" / "xtb_summary.json").exists(): xb.update({c: xtb_best("oh_sampling_anti", c) for c in ANTI})
E = lambda f: np.array([f(c) for c in cells])
prof = {
    "xtb map frame":                 E(lambda c: H2K * sv[c]["E_xtb"]),
    "xtb best network":              E(lambda c: xb[c][0] if c in xb else np.nan),
    "TZVP-D3 // xtb map frame":      E(lambda c: H2K * r3[c]["1_sp_tzvp"]),
    "TZVP-D3 // 3c, one network":    E(lambda c: H2K * r3[c]["4_sp_opt"]),
    # lowest structure known per cell: the six sampled ones and the relaxed map-frame structure of DFT r3
    # (in (90,70) the r3 structure is lower than all six: the xtb ranking does not carry over to DFT there)
    "3c opt, best of 6":             E(lambda c: H2K * min(oh[c]["3_opt_3c"]["min"], r3[c]["3_opt_3c"]) if c in oh else np.nan),
    "TZVP-D3 // 3c, best of 6":      E(lambda c: H2K * min(oh[c]["4_sp_opt"]["min"], r3[c]["4_sp_opt"]) if c in oh else np.nan),
}
for c in cells:
    if c in oh and r3[c]["4_sp_opt"] < oh[c]["4_sp_opt"]["min"]: print(f"note: in {c} the r3 map-frame structure is {H2K * (oh[c]['4_sp_opt']['min'] - r3[c]['4_sp_opt']):.2f} kcal/mol below the six sampled ones (TZVP-D3)")
def bsum(c):
    if c not in oh: return np.nan
    e = H2K * np.array(list(oh[c]["4_sp_opt"]["all"].values()) + [r3[c]["4_sp_opt"]]); return -kT * np.log(np.exp(-(e - e.min()) / kT).sum()) + e.min()
prof["TZVP-D3 // 3c, Boltzmann sum of 6"] = E(bsum)        # over the same seven structures
rel = {k: v - np.nanmin(v[:7]) for k, v in prof.items()}                       # relative to the valley minimum of each level
print("kcal/mol above the valley minimum of each level; valley psi_H =", " ".join(f"{v:6.0f}" for v in x), "| anti (90,70) (60,50)")
for k, v in rel.items(): print(f"{k:34s}", " ".join(f"{e:6.2f}" for e in v[:7]), "|", " ".join(f"{e:6.2f}" for e in v[7:]))
gain = E(lambda c: H2K * oh[c]["4_sp_opt"]["min"] if c in oh else np.nan) - prof["TZVP-D3 // 3c, one network"]
print(f"{'sampling gain at TZVP-D3 (abs.)':34s}", " ".join(f"{e:6.2f}" for e in gain[:7]), "|", " ".join(f"{e:6.2f}" for e in gain[7:]))

T = Template(str(mc.HERE / "maltose_a_84_-145.xyz")); rng = np.random.default_rng(1)
pp, w, Em = mc.load_map("scan2_10deg.dat"); M = mc.precompute(T, pp, rng); M[:, :, :3, 3] /= 10
pH = psiH(pp[:, 1]); syn = np.abs(pH) <= 90; ref = cells.index("100,-150")
def stats(EE, p_fix=None):
    ww = np.exp(-(EE - EE.min()) / kT)
    if p_fix is not None: ww[~syn] *= p_fix / (1 - p_fix) * ww[syn].sum() / ww[~syn].sum()
    ww /= ww.sum(); r = mc.ris_exact(M, ww, T, nlist=(100,))
    ws = ww * syn; mpsi = np.degrees(np.arctan2((ws * np.sin(np.radians(pH))).sum(), (ws * np.cos(np.radians(pH))).sum()))
    return dict(Cinf=float(r["Cinf"]), period=float(r["period_bonds"]), c1=float(r["corr"][1]), c2=float(r["corr"][2]), p_anti=float(ww[~syn].sum()), mean_psiH=float(mpsi))
base = stats(Em)
out = {"cells": cells, "psi_H": x.tolist(), "profiles": {k: [None if np.isnan(e) else float(e) for e in v] for k, v in rel.items()},
       "gain_tzvp": [None if np.isnan(e) else float(e) for e in gain], "map": base, "corrected": {}}
print("\nmap as computed                   :", {k: round(v, 2) for k, v in base.items()})
for k in list(prof)[1:]:
    d = (prof[k] - prof[k][ref]) - (prof["xtb map frame"] - prof["xtb map frame"][ref])     # level minus xtb map frame, relative to (100,-150)
    row = {}
    E2 = Em + np.where(syn, np.interp(np.clip(pH, x.min(), x.max()), x, d[:7]), 0.0)
    if not np.isnan(d[7:]).any():
        E3 = E2 + np.where(~syn, np.where(pp[:, 0] > 75, d[7], d[8]), 0.0); row["anti shifted"] = stats(E3)
        row["anti shift (90,70), (60,50)"] = [float(d[7]), float(d[8])]
    row["p fixed"] = stats(E2, p_fix=base["p_anti"]); out["corrected"][k] = row
    for q in ("anti shifted", "p fixed"):
        if q in row: print(f"{k:34s}: {q:12s}", {a: round(b, 2) for a, b in row[q].items()})
json.dump(out, open(R / "oh_compare.json", "w"), indent=1)

# ---- LaTeX tables for the SI
names = {"valley": "syn valley, $\\varphi=100^\\circ$", "phi wall": "$\\varphi$ wall, $\\psi=-150^\\circ$", "band flip": "band-flip minima",
         "barrier X": "barrier X ($\\psi_H \\approx -120^\\circ$)", "barrier Y": "barrier Y ($\\psi_H \\approx +100^\\circ$)", "secondary": "secondary minima"}
regions = {"valley": VAL, "phi wall": ["80,-150", "110,-150"], "band flip": ANTI, "barrier X": ["100,120", "100,130", "100,140"],
           "barrier Y": ["100,-30", "100,-20", "100,-10"], "secondary": ["160,-110", "60,-170", "170,-60"]}
lev = [lambda c: r3[c]["1_sp_tzvp"], lambda c: r3[c]["2_sp_svp"], lambda c: sv[c]["E_dft"], lambda c: r3[c]["4_sp_opt"], lambda c: sv[c]["E_xtb"]]
mins = [min(f(c) for c in r3) for f in lev]; L = []
for reg, ks in regions.items():
    for i, c in enumerate(ks):
        p_, s_ = c.split(",")
        L.append(f"{names[reg] if i == 0 else ''} & $({p_},{s_})$ & ${psiH(int(s_)):.0f}$ & " + " & ".join(f"{(f(c) - m) * H2K:.1f}" for f, m in zip(lev, mins)) + " \\\\")
(TEX / "tab_dft_benchmark.tex").write_text("\n".join(L) + "\n\\bottomrule\n")
lab = {"xtb map frame": "GFN2-xTB/ALPB, map frame", "xtb best network": "GFN2-xTB/ALPB, lowest of 300 networks",
       "TZVP-D3 // xtb map frame": "r2SCAN-D3/TZVP on the map frame", "TZVP-D3 // 3c, one network": "r2SCAN-D3/TZVP // r2SCAN-3c, map-frame network",
       "3c opt, best of 6": "r2SCAN-3c, lowest sampled", "TZVP-D3 // 3c, best of 6": "r2SCAN-D3/TZVP // r2SCAN-3c, lowest sampled",
       "TZVP-D3 // 3c, Boltzmann sum of 6": "same, Boltzmann sum over the sampled structures"}
f2 = lambda e: "---" if np.isnan(e) else f"{e:.2f}"
(TEX / "tab_oh_valley.tex").write_text("\n".join(f"{lab[k]} & " + " & ".join(f2(e) for e in v) + " \\\\" for k, v in rel.items()) + "\n\\bottomrule\n")
rows = [f"ALPB map as computed & ${base['mean_psiH']:.0f}^\\circ$ & {100*base['p_anti']:.0f}\\% & {base['Cinf']:.2f} & {base['period']:.1f} & {base['c1']:.2f} & {base['c2']:.2f} \\\\"]
for k, row in out["corrected"].items():
    for q in ("anti shifted", "p fixed"):
        if q in row:
            s = row[q]; rows.append(f"{lab[k]}; {'$p$ fixed' if q == 'p fixed' else q} & ${s['mean_psiH']:.0f}^\\circ$ & {100*s['p_anti']:.{0 if s['p_anti'] >= 0.01 else 1}f}\\% & {s['Cinf']:.2f} & {s['period']:.1f} & {s['c1']:.2f} & {s['c2']:.2f} \\\\")
(TEX / "tab_oh_map.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
print("anti cells sampled at DFT level:", have_anti, "| tables written")
