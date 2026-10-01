#!/usr/bin/env python3
"""Same conformations, two force fields: preparation (AFC 2026-10-01, "the comparison between FFs should account
for the differences"). MDAnalysis: ~/miniconda3/bin/python.

Takes the sugar-only trajectories of the six-residue chain from the CHARMM36 replicas (every 5th stored frame) and
from the GLYCAM06 replicas (every 2nd), makes the chain whole, centres it in a 12 nm box, and writes every set in
the atom order of BOTH force fields (atoms matched by residue and name; HO2/H2O etc. are the same hydrogens).
Also stores, per frame and linkage, phi, psi and the six dihedrals about the two glycosidic bonds.
    -> ../ff_components/{C,G}frames_{C,G}order.{gro,xtc}, idx_{C,G}.ndx, ../ff_components/{C,G}frames.npz"""
import warnings
from pathlib import Path
import numpy as np, MDAnalysis as mda
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; MD = HERE.parent / "md_guane"; OUT = HERE.parent / "ff_components"; BOX = 120.0
REN = {"HO2": "H2O", "HO3": "H3O", "HO6": "H6O", "HO4": "H4O"}           # CHARMM name -> GLYCAM name


def sugar(gro, sel):
    full = mda.Universe(str(gro)); return mda.Merge(full.select_atoms(sel))


uc = sugar(MD / "rep_6SC" / "start_2ns.gro", "resname AGLC"); ug = sugar(MD / "glycam_SC6" / "sc6.gro", "resname ROH 4GA 0GA")
# key = (chain residue 1..6 counted from the reducing end, GLYCAM-style atom name)
kc = [(a.resid - uc.atoms[0].resid + 1, REN.get(a.name, a.name)) for a in uc.atoms]
r0 = ug.select_atoms("resname 4GA 0GA").resids.min()
kg = [(1 if a.resname == "ROH" else a.resid - r0 + 1, a.name) for a in ug.atoms]
assert sorted(kc) == sorted(kg) and len(set(kc)) == 129, "atom sets of the two force fields do not match"
c_from_g = np.array([kg.index(k) for k in kc]); g_from_c = np.array([kc.index(k) for k in kg])      # index into the other order
ix = lambda keys, r, n: keys.index((r, n))
for tag, keys in (("C", kc), ("G", kg)):
    with open(OUT / f"idx_{tag}.ndx", "w") as f:
        for r in range(1, 7):
            f.write(f"[ r{r} ]\n" + " ".join(str(i + 1) for i, k in enumerate(keys) if k[0] == r) + "\n")


def dih(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2; b1 = b1 / np.linalg.norm(b1); v = b0 - b0 @ b1 * b1; w = b2 - b2 @ b1 * b1
    return np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w))


def run(tag, u, keys, files, stride, other_order):
    o4 = np.array([ix(keys, r, "O4") for r in range(1, 7)]); resof = np.array([k[0] for k in keys]) - 1
    Q = {r: {n: ix(keys, r, n) for n in ("C1", "O5", "C2", "H1", "O4", "C4", "C5", "C3", "H4")} for r in range(1, 7)}
    natW = mda.Writer(str(OUT / f"{tag}frames_{tag}order.xtc"), 129); othW = mda.Writer(str(OUT / f"{tag}frames_{'G' if tag == 'C' else 'C'}order.xtc"), 129)
    ag_nat = u.atoms; ag_oth = u.atoms[other_order]; D = []; first = True; n = 0
    for f in files:
        u.load_new(str(f))
        for ts in u.trajectory[200::stride]:
            X = ts.positions.astype(float); box = ts.dimensions[:3].astype(float)
            p = X[o4]; d = np.diff(p, axis=0); d -= box * np.round(d / box); O4 = np.vstack([p[:1], p[:1] + np.cumsum(d, 0)])
            Y = X - p[resof]; Y -= box * np.round(Y / box); Y += O4[resof]; Y += BOX / 2 - Y.mean(0)
            ts.positions = Y.astype(np.float32); ts.dimensions = [BOX, BOX, BOX, 90, 90, 90]
            if first:
                ag_nat.write(str(OUT / f"{tag}frames_{tag}order.gro")); ag_oth.write(str(OUT / f"{tag}frames_{'G' if tag == 'C' else 'C'}order.gro")); first = False
            natW.write(ag_nat); othW.write(ag_oth); n += 1
            row = []
            for r in range(1, 6):                                   # linkage r: residue r (O4, C4 side) with residue r+1 (C1 side)
                a, b = Q[r], Q[r + 1]; g = lambda q, nm: Y[q[nm]]
                row.append([dih(g(b, e), g(b, "C1"), g(a, "O4"), g(a, "C4")) for e in ("O5", "C2", "H1")] +
                           [dih(g(b, "C1"), g(a, "O4"), g(a, "C4"), g(a, e)) for e in ("C5", "C3", "H4")])
            D.append(row)
    natW.close(); othW.close(); np.savez_compressed(OUT / f"{tag}frames.npz", dihedrals=np.array(D, np.float32)); print(tag, "frames", n, flush=True)


run("C", uc, kc, [MD / "rep_6SC" / f"r{r}" / "md.xtc" for r in (2, 4, 6, 8, 10)], 5, g_from_c)
run("G", ug, kg, [MD / "glycam_SC6" / "run" / f"r{r}" / "md.xtc" for r in (1, 2, 3, 4, 5)], 2, c_from_g)
