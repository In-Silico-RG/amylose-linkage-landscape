#!/usr/bin/env python3
"""Detailed linkage figure from our own coordinates: (a) 2D structure with atom names and the phi/psi
definitions; (b) real syn and band-flip (anti) geometries from the ALPB map, atoms named; (c) a chain built
from the template, all-syn versus one anti linkage, showing the kink. Run with the chem env python."""
import sys, importlib.util
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; CH = HERE.parent / "cheminf"; FIG = HERE.parent / "figuras"
sys.path.insert(0, str(CH))
from chain_builder import Template, SMILES, build_chain, dihedral
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
spec = importlib.util.spec_from_file_location("db", CH / "dft_benchmark.py"); db = importlib.util.module_from_spec(spec); spec.loader.exec_module(db)

T = Template(str(CH / "maltose_a.xyz"))
m = Chem.AddHs(Chem.MolFromSmiles(SMILES))
NAMES = {2: "C1", 22: "C2", 20: "C3", 7: "C4", 4: "C5", 3: "O5", 8: "O4", 9: "C1′", 18: "C2′", 16: "C3′", 14: "C4′", 11: "C5′", 10: "O5′", 15: "O4′", 32: "H4", 33: "H1′"}
PHI = [10, 9, 8, 7]; PSI = [9, 8, 7, 4]; PHIH = [33, 9, 8, 7]; PSIH = [9, 8, 7, 32]
bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()]
COL = {"C": "0.25", "O": "#d62728", "H": "0.75"}

# ---- (a) 2D depiction with names
for a in m.GetAtoms():
    a.SetIntProp("orig", a.GetIdx())
    if a.GetIdx() in NAMES and a.GetSymbol() != "H": a.SetProp("atomLabel", NAMES[a.GetIdx()].replace("′", "'"))
m2 = Chem.RemoveHs(m)
new_of = {a.GetIntProp("orig"): a.GetIdx() for a in m2.GetAtoms()}
AllChem.Compute2DCoords(m2)
d = rdMolDraw2D.MolDraw2DCairo(900, 520)
opts = d.drawOptions(); opts.addAtomIndices = False; opts.bondLineWidth = 2; opts.padding = 0.08; opts.annotationFontScale = 0.9
cols = {10: (0.16, 0.47, 0.84), 9: (0.16, 0.47, 0.84), 8: (0.5, 0.5, 0.5), 7: (0.92, 0.41, 0.2), 4: (0.92, 0.41, 0.2)}
hl_atoms = [new_of[i] for i in cols]; hl_cols = {new_of[i]: c for i, c in cols.items()}
rdMolDraw2D.PrepareAndDrawMolecule(d, m2, highlightAtoms=hl_atoms, highlightAtomColors=hl_cols)
d.FinishDrawing(); open(FIG / "linkage_2d.png", "wb").write(d.GetDrawingText())

# ---- helpers for 3D
def project(X, ref_idx):
    """Orthographic projection onto the best plane of the atoms ref_idx; returns 2D coords and depth."""
    c = X[ref_idx].mean(0); U, S, Vt = np.linalg.svd(X[ref_idx] - c)
    return (X - c) @ Vt[0], (X - c) @ Vt[1], (X - c) @ Vt[2]

def kabsch_align(P, Q, idx):
    pc, qc = P[idx].mean(0), Q[idx].mean(0); H = (P[idx] - pc).T @ (Q[idx] - qc)
    U, S, Vt = np.linalg.svd(H); dd = np.sign(np.linalg.det(Vt.T @ U.T)); R = Vt.T @ np.diag([1, 1, dd]) @ U.T
    return (R @ (P - pc).T).T + qc

def draw_mol(ax, X, title, label_idx, ref_idx, tors):
    x, y, z = project(X, ref_idx); order = np.argsort(z)
    for i, j in bonds:
        ax.plot([x[i], x[j]], [y[i], y[j]], "-", color="0.55", lw=1.2, zorder=1)
    for i in order:
        s = m.GetAtomWithIdx(int(i)).GetSymbol(); ax.plot(x[i], y[i], "o", color=COL[s], ms=(9 if s != "H" else 5) + 3 * (z[i] - z.min()) / (np.ptp(z) + 1e-9), mec="white", mew=0.5, zorder=2)
    offs = {9: (6, 6), 8: (0, -11), 7: (-14, 6), 4: (6, -10), 10: (6, -8), 32: (-4, 9), 33: (6, 6), 2: (-12, -8)}
    for i in label_idx:
        ax.annotate(NAMES[i], (x[i], y[i]), xytext=offs.get(i, (4, 4)), textcoords="offset points", fontsize=7, color="0.1", zorder=3,
                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))
    for a, b in [(PHI[0], PHI[1]), (PHI[1], PHI[2]), (PHI[2], PHI[3])]:
        ax.plot([x[a], x[b]], [y[a], y[b]], "-", color="#2a78d6", lw=3, alpha=0.6, zorder=1.5)
    for a, b in [(PSI[1], PSI[2]), (PSI[2], PSI[3])]:
        ax.plot([x[a], x[b]], [y[a], y[b]], "-", color="#eb6834", lw=3, alpha=0.6, zorder=1.5)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=7.8, loc="left", fontweight="bold")
    ax.text(0.0, -0.02, tors, transform=ax.transAxes, fontsize=7, va="top", color="0.25")

# ---- (b) real geometries from the ALPB scan
syn_atoms, _ = db.frame(100, -150); anti_atoms, _ = db.frame(90, 70)
Xs = np.array([c for _, c in syn_atoms]); Xa = np.array([c for _, c in anti_atoms])
ring = [3, 4, 7, 20, 22, 2]              # reducing ring (C4 side), the one that carries H4
Xa = kabsch_align(Xa, Xs, ring + [8, 9])  # align on that ring and the bridge so the flip shows
def tors(X):
    return (f"φ = {np.degrees(dihedral(*X[PHI])):.0f}°, ψ = {np.degrees(dihedral(*X[PSI])):.0f}°   "
            f"(H convention: φ_H = {np.degrees(dihedral(*X[PHIH])):.0f}°, ψ_H = {np.degrees(dihedral(*X[PSIH])):.0f}°)")

fig = plt.figure(figsize=(11, 8.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.35, wspace=0.10)
axA = fig.add_subplot(gs[0, 0]); axA.imshow(plt.imread(FIG / "linkage_2d.png")); axA.axis("off")
axA.set_title("a  The α(1→4) linkage of amylose (methyl α-maltoside, atoms named)", fontsize=7.8, loc="left", fontweight="bold")
axA.text(0.0, -0.03, "φ = O5′–C1′–O4–C4  (blue, rotation about C1′–O4)      ψ = C1′–O4–C4–C5  (orange, rotation about O4–C4)\n"
         "hydrogen convention used by Kuttel & Naidoo, Hatcher:  φ_H = H1′–C1′–O4–C4,  ψ_H = C1′–O4–C4–H4;   here φ = φ_H + 116°,  ψ = ψ_H − 118°\n"
         "primed atoms: non-reducing ring (carries C1′ of the linkage); unprimed: reducing ring (carries C4 and H4). Chain node = O4.",
         transform=axA.transAxes, fontsize=7, va="top", color="0.25")
axB = fig.add_subplot(gs[0, 1])
draw_mol(axB, Xs, "b  syn linkage (xtb map, 100°, −150°): H4 points toward C1′\n    " + tors(Xs), [9, 8, 7, 4, 10, 32, 33, 2], ring + [8, 9, 10], "")
axC = fig.add_subplot(gs[1, 1])
draw_mol(axC, Xa, "c  band flip, anti-ψ (xtb map, 90°, 70°): ψ turned by ~180°, H4 points away\n    " + tors(Xa), [9, 8, 7, 4, 10, 32, 33, 2], ring + [8, 9, 10], "")

# ---- (d) chains: all syn vs one band flip
axD = fig.add_subplot(gs[1, 0])
N = 12; syn_chain = [(100, -150)] * (N - 1); flip_chain = list(syn_chain); flip_chain[5] = (90, 70)
resS, O4S = build_chain(T, syn_chain); resF, O4F = build_chain(T, flip_chain)
heavy = [k for k, a in enumerate(T.res) if T.sym[a] != "H"]
allS = np.vstack([r[heavy] for r in resS]); allF = np.vstack([r[heavy] for r in resF])
# align the flipped chain on the first six residues, project on the plane of the syn chain's O4 nodes
idx6 = np.arange(6 * len(heavy)); allF = kabsch_align(allF, allS, idx6)
c = O4S.mean(0); U, S, Vt = np.linalg.svd(O4S - c); e1, e2 = Vt[0], Vt[1]
def pr(X): return (X - c) @ e1, (X - c) @ e2
for X, col, lab, z in ((allS, "#2a78d6", "all linkages syn: regular left-handed helix, 7 residues per turn", 1), (allF, "#eb6834", "same chain with linkage 6 flipped (anti-ψ): the kink", 2)):
    x, y = pr(X); ax = axD
    nh = len(heavy)
    for r in range(N):
        sl = slice(r * nh, (r + 1) * nh)
        ax.plot(x[sl], y[sl], ".", color=col, ms=2.5, alpha=0.55, zorder=z)
    xo, yo = pr(np.array([X[r * nh + heavy.index(T.res.index(T.O4p))] for r in range(N)]))
    ax.plot(xo, yo, "-o", color=col, lw=1.6, ms=4, label=lab, zorder=z + 1)
xo, yo = pr(np.array([allF[r * len(heavy) + heavy.index(T.res.index(T.O4p))] for r in range(N)]))
axD.annotate("band flip here", (xo[6], yo[6]), xytext=(15, -25), textcoords="offset points", fontsize=7, color="#eb6834", arrowprops=dict(arrowstyle="->", color="#eb6834", lw=0.8))
axD.set_aspect("equal"); axD.axis("off"); axD.legend(fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, 1.0)); axD.set_title("d  Twelve residues built from the template (O4 nodes joined; dots = ring heavy atoms)", fontsize=7.8, loc="left", fontweight="bold", pad=22)
axD.text(0.0, -0.02, "one anti linkage turns the following residues over by ~180° about the chain: the helix restarts in a new direction", transform=axD.transAxes, fontsize=7, va="top", color="0.25")
for ext in ("png", "pdf"): fig.savefig(FIG / f"fig_linkage.{ext}", dpi=200, bbox_inches="tight")
print("done")
