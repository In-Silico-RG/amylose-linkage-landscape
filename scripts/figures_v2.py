#!/usr/bin/env python3
"""Publication figure set v2 (2026-09-23), all from resultados/*.json, the map files and the template.
Run with the chem env python (RDKit needed):  /media/aldo/Aldo/envs/chem/bin/python scripts/figures_v2.py [F1 F2 F3 F4]"""
import sys, json, importlib.util
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
HERE = Path(__file__).resolve().parent; RES = HERE.parent / "resultados"; FIG = HERE.parent / "figuras"; CH = HERE.parent / "cheminf"
sys.path.insert(0, str(CH))
from chain_builder import Template, SMILES, build_chain, dihedral
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
ma = importlib.util.module_from_spec(importlib.util.spec_from_file_location("ma", CH / "map_analysis.py")); ma.__spec__.loader.exec_module(ma) if False else None
spec = importlib.util.spec_from_file_location("ma", CH / "map_analysis.py"); ma = importlib.util.module_from_spec(spec); spec.loader.exec_module(ma)
spec = importlib.util.spec_from_file_location("db", CH / "dft_benchmark.py"); db = importlib.util.module_from_spec(spec); spec.loader.exec_module(db)
J = lambda n: json.load(open(RES / n))
BLUE, ORANGE, TEAL, GREY, YEL, MAG = "#2a78d6", "#eb6834", "#1baf7a", "0.45", "#eda100", "#e87ba4"
# Experimental C_inf band on the 0.454 nm bond (as measured, flexible chains) and in rigid-ring units:
# ring/bridge flexibility lowers the rigid-ring C_inf by 26-28 % (SI Table S9: SC6 9.0->6.7, SC18 7.4->5.3),
# so rigid-ring values are compared with the band divided by 0.73 (Kimi r6 W1, 2026-09-24).
EXP = (3.9, 5.8); RING = 0.73; EXP_RIG = (round(EXP[0] / RING, 1), round(EXP[1] / RING, 1))
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False})
T = Template(str(CH / "maltose_a.xyz")); m = Chem.AddHs(Chem.MolFromSmiles(SMILES))
NAMES = {2: "C1", 22: "C2", 20: "C3", 7: "C4", 4: "C5", 3: "O5", 8: "O4", 9: "C1′", 18: "C2′", 16: "C3′", 14: "C4′", 11: "C5′", 10: "O5′", 15: "O4′", 32: "H4", 33: "H1′"}
PHI, PSI, PHIH, PSIH = [10, 9, 8, 7], [9, 8, 7, 4], [33, 9, 8, 7], [9, 8, 7, 32]
bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()]; COL = {"C": "0.25", "O": "#d62728", "H": "0.75"}
RING = [3, 4, 7, 20, 22, 2]


def panel(ax, letter, text=""):
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")
    if text: ax.set_title(text, fontsize=8, loc="left")


def structure_2d(path):
    for a in m.GetAtoms():
        a.SetIntProp("orig", a.GetIdx())
        if a.GetIdx() in NAMES and a.GetSymbol() != "H": a.SetProp("atomLabel", NAMES[a.GetIdx()].replace("′", "'"))
    m2 = Chem.RemoveHs(m); new_of = {a.GetIntProp("orig"): a.GetIdx() for a in m2.GetAtoms()}; AllChem.Compute2DCoords(m2)
    d = rdMolDraw2D.MolDraw2DCairo(900, 520); o = d.drawOptions(); o.bondLineWidth = 2; o.padding = 0.06
    cols = {10: (0.16, 0.47, 0.84), 9: (0.16, 0.47, 0.84), 8: (0.5, 0.5, 0.5), 7: (0.92, 0.41, 0.2), 4: (0.92, 0.41, 0.2)}
    rdMolDraw2D.PrepareAndDrawMolecule(d, m2, highlightAtoms=[new_of[i] for i in cols], highlightAtomColors={new_of[i]: c for i, c in cols.items()})
    d.FinishDrawing(); open(path, "wb").write(d.GetDrawingText())


def project(X, ref):
    c = X[ref].mean(0); U, S, Vt = np.linalg.svd(X[ref] - c); return (X - c) @ Vt[0], (X - c) @ Vt[1], (X - c) @ Vt[2]


def kabsch_align(P, Q, idx):
    pc, qc = P[idx].mean(0), Q[idx].mean(0); H = (P[idx] - pc).T @ (Q[idx] - qc); U, S, Vt = np.linalg.svd(H)
    dd = np.sign(np.linalg.det(Vt.T @ U.T)); R = Vt.T @ np.diag([1, 1, dd]) @ U.T; return (R @ (P - pc).T).T + qc


def draw_mol(ax, X, labels, ref):
    x, y, z = project(X, ref)
    for i, j in bonds: ax.plot([x[i], x[j]], [y[i], y[j]], "-", color="0.6", lw=1.0, zorder=1)
    for i in np.argsort(z):
        s = m.GetAtomWithIdx(int(i)).GetSymbol(); ax.plot(x[i], y[i], "o", color=COL[s], ms=(7.5 if s != "H" else 4) + 2.5 * (z[i] - z.min()) / (np.ptp(z) + 1e-9), mec="white", mew=0.4, zorder=2)
    for a, b in [(10, 9), (9, 8), (8, 7)]: ax.plot([x[a], x[b]], [y[a], y[b]], "-", color=BLUE, lw=3, alpha=0.55, zorder=1.5)
    for a, b in [(8, 7), (7, 4)]: ax.plot([x[a], x[b]], [y[a], y[b]], "-", color=ORANGE, lw=3, alpha=0.55, zorder=1.5)
    offs = {9: (6, 6), 8: (0, -10), 7: (-14, 6), 4: (6, -10), 10: (6, -8), 32: (-4, 9), 33: (6, 6), 2: (-12, -8)}
    for i in labels: ax.annotate(NAMES[i], (x[i], y[i]), xytext=offs.get(i, (4, 4)), textcoords="offset points", fontsize=6.5, zorder=3, bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.3))
    ax.set_aspect("equal"); ax.axis("off")


def tors_text(X):
    return f"φ = {np.degrees(dihedral(*X[PHI])):.0f}°, ψ = {np.degrees(dihedral(*X[PSI])):.0f}°  (ψ$_H$ = {np.degrees(dihedral(*X[PSIH])):.0f}°)"


def F1():
    structure_2d(FIG / "linkage_2d.png")
    fig = plt.figure(figsize=(7.2, 8.2)); gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 0.9, 1.05], hspace=0.38, wspace=0.28)
    ax = fig.add_subplot(gs[0, :]); ax.imshow(plt.imread(FIG / "linkage_2d.png")); ax.axis("off")
    panel(ax, "a", "the α(1→4) linkage of amylose (methyl α-maltoside): φ = O5′–C1′–O4–C4 (blue), ψ = C1′–O4–C4–C5 (orange)")
    ax.text(0, -0.03, "unprimed ring: reducing side (carries C4 and H4); primed ring: non-reducing side (carries C1′). Hydrogen convention of Kuttel–Naidoo and Hatcher:\n"
            "φ$_H$ = H1′–C1′–O4–C4, ψ$_H$ = C1′–O4–C4–H4 (here φ = φ$_H$ + 116°, ψ = ψ$_H$ − 118°). The next residue starts at O4′; the chain of O4 atoms is the virtual chain.",
            transform=ax.transAxes, fontsize=6.3, va="top", color="0.25")
    syn, _ = db.frame(100, -150); anti, _ = db.frame(90, 70); Xs = np.array([c for _, c in syn]); Xa = kabsch_align(np.array([c for _, c in anti]), Xs, RING + [8, 9])
    a1 = fig.add_subplot(gs[1, :2]); sub = a1.get_subplotspec().subgridspec(1, 2, wspace=0.02); a1.remove()
    b1 = fig.add_subplot(sub[0]); draw_mol(b1, Xs, [9, 8, 7, 4, 10, 32, 33], RING + [8, 9, 10]); panel(b1, "b", "syn (100°, −150°)")
    b1.text(0.02, 0.0, tors_text(Xs) + "\nH4 points toward C1′", transform=b1.transAxes, fontsize=6.3, va="top", color="0.25")
    b2 = fig.add_subplot(sub[1]); draw_mol(b2, Xa, [9, 8, 7, 4, 10, 32, 33], RING + [8, 9, 10]); b2.set_title("band flip (90°, 70°)", fontsize=8, loc="left")
    b2.text(0.02, 0.0, tors_text(Xa) + "\nH4 points away; ring turned over", transform=b2.transAxes, fontsize=6.3, va="top", color="0.25")
    # b3: chain with one flip
    b3 = fig.add_subplot(gs[1, 2]); N = 12; sc = [(100, -150)] * (N - 1); fc = list(sc); fc[5] = (90, 70)
    resS, O4S = build_chain(T, sc); resF, O4F = build_chain(T, fc); heavy = [k for k, a in enumerate(T.res) if T.sym[a] != "H"]
    allS = np.vstack([r[heavy] for r in resS]); allF = kabsch_align(np.vstack([r[heavy] for r in resF]), allS, np.arange(6 * len(heavy)))
    c = O4S.mean(0); U, S_, Vt = np.linalg.svd(O4S - c); pr = lambda X: ((X - c) @ Vt[0], (X - c) @ Vt[1]); io = heavy.index(T.res.index(T.O4p)); nh = len(heavy)
    for X, col, z in ((allS, BLUE, 1), (allF, ORANGE, 2)):
        x, y = pr(X); b3.plot(x, y, ".", color=col, ms=1.5, alpha=0.4, zorder=z); xo, yo = pr(np.array([X[r * nh + io] for r in range(N)])); b3.plot(xo, yo, "-o", color=col, lw=1.3, ms=3, zorder=z + 1)
    xo, yo = pr(np.array([allF[r * nh + io] for r in range(N)])); b3.annotate("flip at\nlinkage 6", (xo[6], yo[6]), xytext=(-30, 25), textcoords="offset points", fontsize=6.3, color=ORANGE, arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.7))
    b3.set_aspect("equal"); b3.axis("off"); b3.set_title("12 residues: all syn vs one flip", fontsize=8, loc="left", pad=14)
    for j, (tag, f, mean, letter) in enumerate([("vacuum", "scan_vac_10deg.dat", (112, -113), "c"), ("GBSA water", "scan_gbsa_10deg.dat", (94, -120), ""), ("ALPB water (map used)", "scan2_10deg.dat", (102, -150), "")]):
        a = fig.add_subplot(gs[2, j]); phi, psi, E = ma.load(str(CH / f)); P, S, G = ma.grid(phi, psi, E)
        im = a.imshow(np.ma.masked_invalid(G.T), origin="lower", extent=[-185, 175, -185, 175], vmin=0, vmax=10, cmap="viridis", aspect="equal")
        a.contour(P, S, np.nan_to_num(G.T, nan=50), levels=[1, 2, 4, 6, 8], colors="w", linewidths=0.4, alpha=0.7)
        if j == 2:
            for sysn, col, lab in (("SC18", YEL, "MD, single DP 18"), ("DC6", MAG, "MD, duplex")):
                pp = np.load(CH / f"phipsi_{sysn}.npy"); a.scatter(pp[::10, 0], pp[::10, 1], s=3, c=col, alpha=0.6, lw=0, label=lab)
            a.plot(84, -145, "x", color="white", ms=6, mew=1.6, label="B-amylose crystal"); leg = a.legend(fontsize=5.5, loc="upper left", frameon=True, facecolor="0.25", edgecolor="none"); [t.set_color("white") for t in leg.get_texts()]
        a.plot(*mean, "o", color="white", ms=5, mfc="none", mew=1.4)
        if j == 0:
            a.text(28, -120, "syn\nwell", fontsize=6.3, color="white", va="center"); a.text(28, 120, "band\nflip", fontsize=6.3, color="white", va="center")
        a.set_title(tag, fontsize=8, loc="left"); a.set_xticks([-180, -90, 0, 90, 180]); a.set_yticks([-180, -90, 0, 90, 180]); a.tick_params(labelsize=6)
        a.set_xlabel("φ (°)", fontsize=7)
        if j == 0: a.set_ylabel("ψ (°)", fontsize=7); panel(a, letter)
    cb = fig.colorbar(im, ax=fig.axes[-3:], fraction=0.02, pad=0.02); cb.set_label("E (kcal/mol)", fontsize=7); cb.ax.tick_params(labelsize=6)
    fig.text(0.5, 0.01, "c  three GFN2-xTB maps of the same linkage. Circle: Boltzmann mean of the syn population at 303 K; it slides along the flat valley with solvation.\nThe band-flip well lies 2.9 (vacuum), 1.1 (GBSA) and 1.7 (ALPB) kcal/mol above the syn minimum; white = rejected cells (ring deformed).", ha="center", fontsize=6.5, color="0.25")
    for ext in ("png", "pdf"): fig.savefig(FIG / f"F1_linkage_map.{ext}", dpi=250, bbox_inches="tight")
    print("F1 done")




def F2():
    """One step to the chain: correlations; the built helix; rigid vs flexible ring."""
    rf = J("ris_residue_frame.json"); mp = J("ris_exact_maps.json")["scan2_10deg.dat"]; mdt = J("ris_exact_mdtorsions.json")
    fig = plt.figure(figsize=(7.2, 5.2)); gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.85], hspace=0.6, wspace=0.45)
    for j, (s_, col, lab) in enumerate((("SC6", BLUE, "single chain, DP 6"), ("SC18", TEAL, "single chain, DP 18"), ("DC6", "#008300", "duplex, DP 6"))):
        ax = fig.add_subplot(gs[0, j]); d = J(f"{s_}_O4.json"); k = np.arange(len(d["corr"]))
        ax.errorbar(k, d["corr"], yerr=d["corr_se"], fmt="o", color=col, ms=3, lw=0.7, label="MD (block errors)")
        R = np.array(rf[s_]["R_mean"]); v = np.array(rf[s_]["v_mean_frame"]); l2 = rf[s_]["l2"]
        ax.plot(k, [(v @ np.linalg.matrix_power(R, i) @ v) / l2 for i in k], "-", color="0.15", lw=1.1, label="one step of this run (RIS)")
        kk = np.arange(min(8, len(k))); ax.plot(kk, mp["corr"][:len(kk)], "--", color=ORANGE, lw=1.1, label="xtb map (exact RIS)")
        ax.axhline(0, color="0.85", lw=0.5, zorder=0); ax.set_xlabel("bonds apart, k"); ax.set_title(lab, fontsize=8, loc="left")
        r = rf[s_]["residue_frame"]; ax.text(0.97, 0.95, f"period {r['period_bonds']:.1f} bonds\nhelix memory r = {r['lam_helix']:.2f}\naxis memory ρ = {r['lam_real'][0]:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=6.3, color="0.3")
        if j == 0: ax.set_ylabel("tangent correlation ⟨t$_i$·t$_{i+k}$⟩"); panel(ax, "a"); ax.legend(fontsize=5.8, loc="lower left")
    # b: built helix from the map's syn mean, coloured by the decay r^k
    ax = fig.add_subplot(gs[1, :2]); N = 16; res, O4 = build_chain(T, [(102, -150)] * (N - 1)); heavy = [k for k, a in enumerate(T.res) if T.sym[a] != "H"]
    c = O4.mean(0); U, S_, Vt = np.linalg.svd(O4 - c); pr = lambda X: ((X - c) @ Vt[0], (X - c) @ Vt[1])
    rr = mp["lam"][0] if isinstance(mp["lam"], list) else 0.81
    for i, rres in enumerate(res):
        x, y = pr(rres[heavy]); a_ = max(0.08, 0.81 ** i); ax.plot(x, y, ".", color=BLUE, ms=3, alpha=a_, zorder=2)
    xo, yo = pr(O4); ax.plot(xo, yo, "-", color="0.3", lw=1, alpha=0.6, zorder=1)
    for i in range(N): ax.plot(xo[i], yo[i], "o", color=BLUE, ms=4, alpha=max(0.08, 0.81 ** i), zorder=3)
    ax.set_aspect("equal"); ax.axis("off"); panel(ax, "b", "the residual helix read from the map's transfer matrix")
    ax.text(0.0, -0.02, "rigid chain at the map's syn mean (102°, −150°): left-handed, 6.8 residues per turn.\nFading = the real chain's phase memory r$^k$ (r = 0.81): 23 % after one turn, 5 % after two.",
            transform=ax.transAxes, fontsize=6.3, va="top", color="0.25")
    # c: rigid vs flexible ring bars
    ax = fig.add_subplot(gs[1, 2]); labels = ["c$_1$", "c$_2$", "c$_3$"]; x = np.arange(3); w = 0.2
    series = [("xtb map, rigid", mp["corr"][1:4], ORANGE), ("MD torsions on rigid template", mdt["SC18"]["corr"][1:4], "0.55"),
              ("MD, flexible residues", rf["SC18"]["residue_frame"]["corr"][1:4], TEAL), ("MD, measured", J("SC18_O4.json")["corr"][1:4], "0.15")]
    for i, (lab, vals, col) in enumerate(series): ax.bar(x + (i - 1.5) * w, vals, w, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("correlation"); ax.set_ylim(0, 1.0); ax.legend(fontsize=5.3, loc="upper right", ncol=1); panel(ax, "c", "the rigid ring, isolated (DP 18)")
    ax.text(0.0, -0.22, "same torsions, rigid template → the map's values;\nflexible residues → the measured ones (−30 % in C∞)", transform=ax.transAxes, fontsize=6.0, va="top", color="0.25")
    for ext in ("png", "pdf"): fig.savefig(FIG / f"F2_one_step.{ext}", dpi=250, bbox_inches="tight")
    print("F2 done")




def F3():
    """Long chains: C_n(N), <R2>/<Rg2>, and the persistence-length window with a drawing of the window."""
    from scipy.optimize import minimize_scalar
    rf = J("ris_residue_frame.json"); ex = J("ris_exact_maps.json")
    fig = plt.figure(figsize=(7.2, 3.0)); gs = fig.add_gridspec(1, 3, wspace=0.45)
    # a: C_n(N)
    ax = fig.add_subplot(gs[0, 0])
    for big, small, key, lab, col in (("mc_scan2_10deg_big.json", "mc_scan2_10deg.json", "scan2_10deg.dat", "map as computed (7 % flips)", BLUE), ("mc_scan2_10deg_Eshift3_big.json", "mc_scan2_10deg_Eshift3.json", "scan2_10deg_Eshift3.dat", "band flips at 1 %", ORANGE)):
        r = {**J(small)["results"], **J(big)["results"]}; N = sorted(int(n) for n in r)
        ax.errorbar(N, [r[str(n)]["Cn"] for n in N], yerr=[r[str(n)]["Cn_se"] for n in N], fmt="o-", color=col, ms=2.5, lw=1, label=lab); ax.axhline(ex[key]["Cinf"], color=col, ls=":", lw=0.8)
    ev = ([18, 40, 100, 200, 400], [5.76, 7.00, 8.42, 9.45, 10.41]); ax.plot(*ev, "^--", color=BLUE, ms=3, lw=0.9, mfc="white", label="+ excluded volume, 0.25 nm")
    ax.axhspan(*EXP_RIG, color="0.9", zorder=0); ax.text(7, EXP_RIG[1] - 0.12, "experiment, rigid-ring units", fontsize=5.5, color="0.4", va="top")
    for y in EXP: ax.axhline(y, color="0.55", ls="--", lw=0.5, zorder=0)
    ax.text(300, EXP[0] + 0.15, "as measured (for MD)", fontsize=5, color="0.45", ha="center", va="bottom")
    for s_ in ["SC6", "SC12", "SC18", "SC24", "DC6"]:
        d = J(f"{s_}_O4.json"); ax.plot(d["N"], d["Cn"], "D", color="0.3", ms=3.5, zorder=5)
    ax.text(5.5, 2.3, "MD, 10 ns (not converged)", fontsize=6, color="0.3")
    for n_, key in ((6, "SC6_charmm_rep"), (12, "SC12_charmm_rep")):          # five 100-ns replicas (2026-09-30)
        g = J(f"replica_{key}.json")["group"]["Cn_meas"]; ax.errorbar([n_], [g["mean"]], yerr=[g["se"]], fmt="D", color=TEAL, ms=4, mec="k", mew=0.5, capsize=1.5, zorder=6)
    ax.text(40, 3.2, "MD, 5 × 100 ns", fontsize=6, color=TEAL, va="center")
    ax.set_xscale("log"); ax.set_xlabel("residues N"); ax.set_ylabel("$C_n$"); ax.set_ylim(2, 14.5); ax.legend(fontsize=5.6, loc="upper left"); panel(ax, "a", "MC chains from the map")
    # b: shape ratio and nu
    ax = fig.add_subplot(gs[0, 1]); r = {**J("mc_scan2_10deg.json")["results"], **J("mc_scan2_10deg_big.json")["results"]}; N = sorted(int(n) for n in r)
    ax.plot(N, [r[str(n)]["R2"] / r[str(n)]["Rg2"] for n in N], "o-", color=BLUE, ms=2.5, lw=1, label="⟨R²⟩/⟨R$_g$²⟩"); ax.axhline(6, color="0.6", ls=":", lw=0.8); ax.text(7, 6.1, "Gaussian value 6", fontsize=6, color="0.4")
    ax2 = ax.twinx(); nu = [(n, r[str(n)]["nu_local"]) for n in N if r[str(n)]["nu_local"] is not None]
    ax2.plot([n for n, _ in nu], [v for _, v in nu], "s-", color=ORANGE, ms=2.5, lw=1, label="local exponent ν"); ax2.axhline(0.5, color=ORANGE, ls=":", lw=0.7); ax2.set_ylim(0.4, 0.9); ax2.set_ylabel("ν", color=ORANGE); ax2.tick_params(axis="y", colors=ORANGE); ax2.spines["right"].set_visible(True); ax2.spines["right"].set_color(ORANGE)
    ax.set_xscale("log"); ax.set_xlabel("residues N"); ax.set_ylabel("⟨R²⟩/⟨R$_g$²⟩"); ax.set_ylim(5.5, 8); panel(ax, "b", "Gaussian beyond DP 400")
    # c: lp vs window with a window drawing
    ax = fig.add_subplot(gs[0, 2]); Smax = 80
    for s_, col, lab in (("DC6", "#008300", "duplex (one-step RIS)"), ("SC6", BLUE, "single, DP 6 (RIS)"), ("SC18", TEAL, "single, DP 18 (RIS)")):
        R = np.array(rf[s_]["R_mean"]); v = np.array(rf[s_]["v_mean_frame"]); l2 = rf[s_]["l2"]; l = rf[s_]["l"]; c = np.array([(v @ np.linalg.matrix_power(R, i) @ v) / l2 for i in range(Smax + 1)])
        Sx = np.arange(2, Smax + 1); ax.plot(Sx, [minimize_scalar(lambda x: np.sum((np.exp(-np.arange(S + 1) * l / x) - c[:S + 1]) ** 2), bounds=(0.01, 500), method="bounded").x for S in Sx], "-", color=col, lw=1.2, label=lab)
        d = J(f"{s_}_O4.json"); ax.plot(d["n"], d["lp_tangent"], "o", color=col, ms=4.5, mfc="white", mew=1.2)
    lp = J("mc_scan2_10deg_big.json")["results"]["1000"]["lp_KP"]; ax.plot([int(k) for k in lp], list(lp.values()), "s-", color="0.35", ms=3, lw=0.9, label="MC chain, DP 1000")
    ax.set_xscale("log"); ax.set_xlabel("fit window S (bonds)"); ax.set_ylabel("Kratky–Porod $l_p$ (nm)"); ax.set_xticks([2, 5, 10, 20, 50]); ax.set_xticklabels(["2", "5", "10", "20", "50"]); ax.legend(fontsize=5.4, loc="upper left"); panel(ax, "c", "$l_p$ vs fit window")
    # window drawing inset: a wavy chain with a bracket
    ins = ax.inset_axes([0.56, 0.16, 0.42, 0.22]); t = np.linspace(0, 4 * np.pi, 60); ins.plot(t, 0.5 * np.sin(t), "-", color="0.3", lw=1); ins.plot(t[::6], 0.5 * np.sin(t[::6]), "o", color="0.3", ms=2)
    ins.plot([0, 2 * np.pi], [-0.95, -0.95], "-", color=ORANGE, lw=1.2); ins.text(np.pi, -1.25, "window S", fontsize=5.5, color=ORANGE, ha="center", va="top"); ins.set_ylim(-1.8, 1); ins.axis("off")
    ax.plot([], [], "o", color="0.4", mfc="white", mew=1.2, ms=4.5, label="MD run, fitted over its own length"); ax.legend(fontsize=5.2, loc="upper left")
    for ext in ("png", "pdf"): fig.savefig(FIG / f"F3_long_chains.{ext}", dpi=250, bbox_inches="tight")
    print("F3 done")


def F4():
    """The two numbers of the map and the plane."""
    fig = plt.figure(figsize=(7.2, 3.0)); gs = fig.add_gridspec(1, 3, wspace=0.42)
    ax = fig.add_subplot(gs[0, 0]); rows = J("cinf_vs_synvalley.json"); x = np.array([r["mean_psiH"] for r in rows]); c = np.array([r["Cinf"] for r in rows]); o = np.argsort(x); x, c = x[o], c[o]
    ax.plot(x, c, "-", color="0.15", lw=1.5); ax.axhspan(*EXP_RIG, color="0.9", zorder=0)
    for xv, lab, col, mk in ((-32, "xtb/ALPB", BLUE, "o"), (-2, "xtb/GBSA", BLUE, "s"), (5, "xtb, vacuum", BLUE, "^"), (-26, "CHARMM36 MD", TEAL, "D"), (-30, "GLYCAM06 MD", ORANGE, "D"), (-35, "CSFF/TIP3P PMF", ORANGE, "o"), (-25, "CSFF vacuum", ORANGE, "s"), (-30, "CHARMM36 map", TEAL, "^"), (-27, "B-amylose crystal", "0.4", "x")):
        yv = np.interp(xv, x, c); ax.plot(xv, yv, mk, color=col, ms=5, mfc=("white" if mk != "x" else col), mew=1.3, label=lab)
    oc = J("oh_compare.json")["corrected"]; best = lambda k: oc[k].get("anti shifted", oc[k]["p fixed"])     # hydroxyl networks sampled (2026-09-30)
    dx, dd = best("xtb best network"), best("TZVP-D3 // 3c, best of 6")
    ax.plot(dx["mean_psiH"], dx["Cinf"], "v", color=BLUE, ms=5.5, mfc=BLUE, mew=1.0, label="xtb, OH sampled")
    ax.plot(dd["mean_psiH"], dd["Cinf"], "^", color="k", ms=5.5, mfc="k", mew=1.0, label="DFT, OH sampled")
    ax.set_yscale("log"); ax.set_ylim(0.4, 60); ax.set_yticks([0.5, 1, 2, 5, 10, 20, 50]); ax.set_yticklabels(["0.5", "1", "2", "5", "10", "20", "50"]); ax.invert_xaxis()
    ax.set_xlabel("mean ψ$_H$ of the syn population (°)"); ax.set_ylabel("$C_\\infty$ (rigid ring)"); ax.legend(fontsize=4.7, loc="upper left", labelspacing=0.15, handletextpad=0.3, borderaxespad=0.1, markerscale=0.85); panel(ax, "a", "first number: valley position")
    ax.text(0.98, 0.03, "grey: experiment,\nrigid-ring units", transform=ax.transAxes, fontsize=5.5, ha="right", va="bottom", color="0.4")
    ax = fig.add_subplot(gs[0, 1]); rows = J("cinf_vs_antipsi.json"); p = np.array([r["p_anti"] for r in rows]) * 100; c = np.array([r["Cinf"] for r in rows])
    ax.plot(p, c, "-", color="0.15", lw=1.5); ax.axhspan(*EXP_RIG, color="0.9", zorder=0)
    for xv, lab, col, mk in ((7.4, "xtb/ALPB", BLUE, "o"), (6.3, "CHARMM36 MD, this work", TEAL, "D"), (7.6, "CHARMM36 map, Lutsyk", TEAL, "s"), (1.06, "CSFF/TIP3P PMF", ORANGE, "o"), (0.75, "GLYCAM06 map, Lutsyk", ORANGE, "s"), (4.0, "GLYCAM06 MD, Sattelle", ORANGE, "D"), (0.7, "GLYCAM06 MD, this work", ORANGE, "v")):
        yv = np.interp(xv, p[::-1], c[::-1]); ax.plot(xv, yv, mk, color=col, ms=5, mfc="white", mew=1.3, label=lab)
    ax.set_xscale("log"); ax.set_xlim(0.1, 50); ax.set_xticks([0.1, 0.3, 1, 3, 10, 30]); ax.set_xticklabels(["0.1", "0.3", "1", "3", "10", "30"]); ax.set_ylim(4, 12)
    ax.set_xlabel("band-flip population (%)"); ax.set_ylabel("$C_\\infty$ (rigid ring)"); ax.legend(fontsize=5.2, loc="lower left", framealpha=0.9); panel(ax, "b", "second number: band flips")
    ax = fig.add_subplot(gs[0, 2]); rows = J("cinf_grid_psi_p.json"); ps = sorted(set(q["p"] for q in rows)); ks = sorted(set(q["k"] for q in rows))
    X = np.array([[next(q["mean_psiH"] for q in rows if q["k"] == k and q["p"] == pp) for pp in ps] for k in ks]); Y = np.array([[pp * 100 for pp in ps] for k in ks]); Z = np.array([[next(q["Cinf"] for q in rows if q["k"] == k and q["p"] == pp) for pp in ps] for k in ks])
    ax.contourf(X, Y, np.log10(Z), levels=np.linspace(0, 1.6, 17), cmap="Blues"); cs = ax.contour(X, Y, Z, levels=[2, 3, 4, 6, 8, 12, 20], colors="0.2", linewidths=0.5); ax.clabel(cs, fmt="%g", fontsize=5.5)
    ax.contour(X, Y, Z, levels=list(EXP_RIG), colors=ORANGE, linewidths=1.3)
    ax.plot(-32, 7.4, "o", color="white", mec="k", ms=6, label="xtb/ALPB"); ax.plot(-2, 8.0, "s", color="white", mec="k", ms=6, label="xtb/GBSA"); ax.plot(-26, 6.3, "D", color=TEAL, mec="k", ms=6, label="CHARMM36 MD"); ax.plot(-30, 0.7, "D", color=ORANGE, mec="k", ms=6, label="GLYCAM06 MD"); ax.plot(dx["mean_psiH"], dx["p_anti"] * 100, "v", color=BLUE, mec="k", ms=6, label="xtb, OH sampled"); ax.plot(dd["mean_psiH"], dd["p_anti"] * 100, "^", color="k", mec="k", ms=6, label="r2SCAN-D3, OH sampled")
    ax.text(-3, 22, "orange: experiment\n÷ 0.73 (rigid ring)", fontsize=5.5, color="0.2", ha="left", va="center", bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.3))
    ax.set_yscale("symlog", linthresh=1); ax.set_ylim(0, 30); ax.set_yticks([0, 1, 3, 10, 30]); ax.set_yticklabels(["0", "1", "3", "10", "30"]); ax.invert_xaxis()
    ax.set_xlabel("mean ψ$_H$ of the syn population (°)"); ax.set_ylabel("band-flip population (%)"); ax.legend(fontsize=4.9, loc="lower right", frameon=True, framealpha=0.85, edgecolor="none", labelspacing=0.2, borderaxespad=0.2); panel(ax, "c", "the plane (orange: experiment)")
    for ext in ("png", "pdf"): fig.savefig(FIG / f"F4_two_numbers.{ext}", dpi=250, bbox_inches="tight")
    print("F4 done")


if __name__ == "__main__":
    which = sys.argv[1:] or ["F1"]
    for w in which: globals()[w]()
