#!/usr/bin/env python3
"""Same conformations, two force fields: analysis (after ff_components_prep.py and the four GROMACS reruns in
../ff_components/, see ff_components/README). chem env (panedr).

Per linkage (residue r with residue r+1) and frame, energy of the isolated chain split into: explicit torsion terms
about C1'-O4 and about O4-C4 (from the six dihedrals and ff_torsions.json), 1-4 pairs between the two residues
(Lennard-Jones and Coulomb), and the other non-bonded pairs between the two residues (Lennard-Jones, Coulomb).
A: flipped minus normal linkages, per component, each force field on each set of conformations.
B: all pairs of residues that are not neighbours, slope against the number of flipped linkages in the frame.
C: normal linkages only, total of the linkage terms binned in phi and in psi.
    -> ../resultados/ff_components.json; kcal/mol"""
import json
from pathlib import Path
import numpy as np, panedr
HERE = Path(__file__).resolve().parent; W = HERE.parent / "ff_components"; RES = HERE.parent / "resultados"
KJ = 4.184; T = json.loads((RES / "ff_torsions.json").read_text())["terms"]; FFN = {"C": "CHARMM36", "G": "GLYCAM06"}
tors = lambda L, th: sum(t["k"] * (1 + np.cos(np.radians(t["n"] * th - t["delta"]))) for t in L)


def load(fr, ff):
    d = panedr.edr_to_df(str(W / f"{fr}frames_{ff}ff.edr")); D = np.load(W / f"{fr}frames.npz")["dihedrals"].astype(float)   # frames, 5 linkages, 6
    n = min(len(d), len(D)); d = d.iloc[:n]; D = D[:n]; tt = T[FFN[ff]]
    c = {}
    c["torsion about C1'-O4"] = sum(tors(tt["phi"][a], D[:, :, j]) for j, a in enumerate(("O5", "C2", "H1")))
    c["torsion about O4-C4"] = sum(tors(tt["psi"][a], D[:, :, 3 + j]) for j, a in enumerate(("C5", "C3", "H4")))
    col = lambda k, r: d[f"{k}:r{r}-r{r+1}"].to_numpy() / KJ
    for name, k in (("1-4 Lennard-Jones", "LJ-14"), ("1-4 Coulomb", "Coul-14"), ("other Lennard-Jones", "LJ-SR"), ("other Coulomb", "Coul-SR")):
        c[name] = np.column_stack([col(k, r) for r in range(1, 6)])
    far = sum(d[f"{k}:r{i}-r{j}"].to_numpy() for k in ("Coul-SR", "LJ-SR") for i in range(1, 7) for j in range(i + 2, 7)) / KJ
    psiH = (D[:, :, 3] + 117.8 + 180) % 360 - 180
    return c, np.abs(psiH) > 90, D[:, :, 0], D[:, :, 3], far


out = {}
for fr in "CG":
    for ff in "CG":
        c, flip, phi, psi, far = load(fr, ff); key = f"{FFN[fr]} conformations, {FFN[ff]} energy"; c["sum of the linkage terms"] = sum(c.values())
        A = {}
        for k, v in c.items():
            a, b = v[flip], v[~flip]; A[k] = dict(diff=float(a.mean() - b.mean()), se=float(np.sqrt(a.var() / len(a) + b.var() / len(b))))
        nf = flip.sum(1); sl = np.polyfit(nf, far, 1)[0]
        tot = c["sum of the linkage terms"]; ok = ~flip
        prof = {}
        for nm, x, edges in (("phi", phi, np.arange(50, 131, 10)), ("psi", np.where(psi > 100, psi - 360, psi), np.arange(-190, -99, 10))):
            m = [float(tot[ok & (x >= lo) & (x < lo + 10)].mean()) if (ok & (x >= lo) & (x < lo + 10)).sum() > 50 else None for lo in edges[:-1]]
            mn = min(v for v in m if v is not None); prof[nm] = dict(centres=[float(lo + 5) for lo in edges[:-1]], E=[None if v is None else round(v - mn, 2) for v in m])
        out[key] = dict(n_flipped=int(flip.sum()), n_normal=int((~flip).sum()), flipped_minus_normal=A, far_pairs_slope_per_flip=float(sl), profiles=prof)
        print("\n" + key, f"(flipped {flip.sum()}, normal {(~flip).sum()})")
        for k, v in A.items(): print(f"   {k:26s} {v['diff']:+7.2f} +- {v['se']:.2f}")
        print(f"   non-neighbour ring pairs, per flip in the frame {sl:+.2f}")
        print("   phi profile", prof["phi"]["E"]); print("   psi profile", prof["psi"]["E"])
(RES / "ff_components.json").write_text(json.dumps(out, indent=1))

# ---- D: what the linkage terms alone predict for the other force field (one-sided free-energy perturbation),
#         and which component steepens the energy away from the centre of the valley
kT = 0.0019872 * 303.15; pred = {}
for fr, other in (("C", "G"), ("G", "C")):
    c0, flip, phi, psi, _ = load(fr, fr); c1, *_ = load(fr, other)
    dU = sum(c1.values()) - sum(c0.values()); w = np.exp(-(dU - dU.mean()) / kT)
    odds = flip.mean() / (1 - flip.mean()); ratio = w[flip].mean() / w[~flip].mean(); o2 = odds * ratio
    # block spread of the estimate (10 blocks of frames)
    bl = [w[i][flip[i]].mean() / w[i][~flip[i]].mean() for i in np.array_split(np.arange(len(w)), 10) if flip[i].sum() > 5]
    psiu = np.where(psi > 100, psi - 360, psi); ok = ~flip
    centre = ok & (phi >= 70) & (phi < 90) & (psiu >= -160) & (psiu < -140)
    regions = {"phi 100-120": ok & (phi >= 100) & (phi < 120), "psi -130 to -110": ok & (psiu >= -130) & (psiu < -110), "psi -190 to -170": ok & (psiu >= -190) & (psiu < -170)}
    steep = {FFN[ff]: {rg: {k: float(v[m].mean() - v[centre].mean()) for k, v in cc.items()} for rg, m in regions.items()} for ff, cc in ((fr, c0), (other, c1))}
    pred[f"{FFN[fr]} conformations"] = dict(flips_as_sampled=float(flip.mean()), flips_predicted_for=FFN[other], flips_predicted=float(o2 / (1 + o2)),
                                             ratio=float(ratio), ratio_block_range=[float(min(bl)), float(max(bl))], dU_sd=float(dU.std()), steepness=steep)
    print(f"\n{FFN[fr]} conformations: flips {100*flip.mean():.2f} % as sampled -> {100*o2/(1+o2):.2f} % predicted for {FFN[other]} from the linkage terms (odds ratio {ratio:.3f}, blocks {min(bl):.3f}-{max(bl):.3f}; sd of dU {dU.std():.1f} kcal/mol)")
    for ff in steep:
        for rg in regions:
            s = steep[ff][rg]; print(f"   {ff:9s} {rg:18s} total {sum(s.values()):+.2f} | " + " ".join(f"{k.split()[0][:4]}{k.split()[-1][:3]} {v:+.2f}" for k, v in s.items()))
out["prediction"] = pred
(RES / "ff_components.json").write_text(json.dumps(out, indent=1))

# ---- LaTeX tables for the SI
TEX = HERE.parent / "manuscript_JPCB"
cols = ["CHARMM36 conformations, CHARMM36 energy", "CHARMM36 conformations, GLYCAM06 energy", "GLYCAM06 conformations, CHARMM36 energy", "GLYCAM06 conformations, GLYCAM06 energy"]
names = {"torsion about C1'-O4": "torsion terms about C1$'$--O4", "torsion about O4-C4": "torsion terms about O4--C4", "1-4 Lennard-Jones": "1--4 pairs, Lennard-Jones",
         "1-4 Coulomb": "1--4 pairs, Coulomb", "other Lennard-Jones": "other pairs, Lennard-Jones", "other Coulomb": "other pairs, Coulomb", "sum of the linkage terms": "sum"}
L = []
for k, lab in names.items():
    if k.startswith("sum"): L.append("\\midrule")
    L.append(lab + " & " + " & ".join(f"${out[c]['flipped_minus_normal'][k]['diff']:+.2f}$" + (f" ({out[c]['flipped_minus_normal'][k]['se']:.2f})" if k.startswith('sum') else "") for c in cols) + " \\\\")
L.append("other pairs of rings, per flip & " + " & ".join(f"${out[c]['far_pairs_slope_per_flip']:+.2f}$" for c in cols) + " \\\\")
(TEX / "tab_ff_flip.tex").write_text("\n".join(L) + "\n\\bottomrule\n")
S = []
for conf in ("CHARMM36 conformations", "GLYCAM06 conformations"):
    st = pred[conf]["steepness"]
    for rg, lab in (("phi 100-120", "$\\varphi$ 100--120$^\\circ$"), ("psi -130 to -110", "$\\psi$ $-130$ to $-110^\\circ$"), ("psi -190 to -170", "$\\psi$ $-190$ to $-170^\\circ$")):
        g = lambda ff, keys: sum(st[ff][rg][k] for k in keys)
        tk, lk, ck = ("torsion about C1'-O4", "torsion about O4-C4"), ("1-4 Lennard-Jones", "other Lennard-Jones"), ("1-4 Coulomb", "other Coulomb")
        S.append(f"{conf.split()[0] if rg.startswith('phi') else ''} & {lab} & " + " & ".join(f"${g(ff, tk + lk + ck):+.2f}$ & ${g(ff, tk):+.2f}$ & ${g(ff, lk):+.2f}$ & ${g(ff, ck):+.2f}$" for ff in ("CHARMM36", "GLYCAM06")) + " \\\\")
    if conf.startswith("CHARMM"): S.append("\\midrule")
(TEX / "tab_ff_steep.tex").write_text("\n".join(S) + "\n\\bottomrule\n")
print("\n".join(L)); print("\n".join(S))
