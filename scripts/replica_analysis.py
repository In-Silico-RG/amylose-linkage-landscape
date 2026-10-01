#!/usr/bin/env python3
"""Analysis of the GUANE runs of 2026-09-25/30 (CHARMM36 replicas 5 x 100 ns of SC6 and SC12, the 200 ns
extensions, GLYCAM06j-1 SC6 replicas): band-flip fraction, residue-frame transfer matrix and C_inf, and the
ring-flexibility factor, per trajectory and with errors from blocks and from the spread between replicas.
Conventions of residue_frame_ris.py and ring_factor_blocks.py (frames from C1, C4, O5; bond O4_i -> O4_{i+1}
in the frame of residue i+1; phi = O5(i+1)-C1(i+1)-O4(i)-C4(i), psi = C1(i+1)-O4(i)-C4(i)-C5(i);
band flip: |psi_H| > 90 with psi_H = psi + 117.8).

Stage A (MDAnalysis: ~/miniconda3/bin/python):   extract TAG TOP TRJ
    TOP: any structure/topology whose atoms match TRJ. For the sugar-only replica xtc, pass the full .gro with
    a trailing '@sugar': the carbohydrate atoms are selected first (the xtc group holds exactly those, in order).
    -> ../resultados/rep/TAG.npz  (t [F], R [F,L,3,3], v [F,L,3], phipsi [F,L,2], R2 [F])
Stage B (RDKit chain builder: chem env):         stats NAME TAG [TAG ...]
    per trajectory, t >= 2 ns, and for the group NAME: mean +- SE over trajectories
    -> ../resultados/replica_<NAME>.json
Table for the SI:                                table   -> ../manuscript_JPCB/tab_replicas.tex
"""
import sys, json
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
RES = HERE.parent / "resultados"; REP = RES / "rep"
SUGAR = "resname AGLC 4GA 0GA"
T0_PS, NBLOCK = 2000.0, 7


def dih(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1, axis=1)[:, None]
    v = b0 - np.sum(b0 * b1, 1)[:, None] * b1; w = b2 - np.sum(b2 * b1, 1)[:, None] * b1
    return np.degrees(np.arctan2(np.sum(np.cross(b1, v) * w, 1), np.sum(v * w, 1)))


def extract(tag, top, trj):
    import MDAnalysis as mda, warnings
    warnings.filterwarnings("ignore")
    if top.endswith("@sugar"):
        full = mda.Universe(top[:-6]); sel = full.select_atoms(SUGAR + " ROH")
        u = mda.Merge(sel); u.load_new(trj)
    else:
        u = mda.Universe(top, trj)
    res = u.select_atoms(SUGAR).residues
    ix = {nm: np.array([r.atoms.select_atoms("name " + nm).indices[0] for r in res]) for nm in ("C1", "C4", "O5", "C5", "O4")}
    T, R, V, PP, R2 = [], [], [], [], []
    for k, ts in enumerate(u.trajectory):
        X = ts.positions.astype(float); box = ts.dimensions[:3].astype(float)
        raw = X[ix["O4"]]; d = np.diff(raw, axis=0); d -= box * np.round(d / box)
        O4 = np.vstack([raw[:1], raw[:1] + np.cumsum(d, axis=0)])           # unwrapped nodes, Angstrom
        P = {}
        for nm in ix:                                                       # residue whole around its O4
            d = X[ix[nm]] - raw; d -= box * np.round(d / box); P[nm] = O4 + d
        if k == 0:
            link = np.linalg.norm(P["C1"][1:] - O4[:-1], axis=1)
            assert link.max() < 1.6, f"C1(i+1)-O4(i) not bonded: {link}"
        e1 = P["C4"] - P["C1"]; e1 /= np.linalg.norm(e1, axis=1)[:, None]
        w = P["O5"] - P["C1"]; e2 = w - np.sum(w * e1, 1)[:, None] * e1; e2 /= np.linalg.norm(e2, axis=1)[:, None]
        F = np.stack([e1, e2, np.cross(e1, e2)], axis=2)                    # columns e1 e2 e3
        T.append(ts.time); R.append(np.einsum("iba,ibc->iac", F[:-1], F[1:]))
        V.append(np.einsum("iba,ib->ia", F[1:], O4[1:] - O4[:-1]) / 10.0)
        PP.append(np.column_stack([dih(P["O5"][1:], P["C1"][1:], O4[:-1], P["C4"][:-1]), dih(P["C1"][1:], O4[:-1], P["C4"][:-1], P["C5"][:-1])]))
        R2.append(np.sum((O4[-1] - O4[0]) ** 2) / 100.0)
    REP.mkdir(exist_ok=True)
    np.savez_compressed(REP / f"{tag}.npz", t=np.array(T), R=np.array(R, np.float32), v=np.array(V, np.float32), phipsi=np.array(PP, np.float32), R2=np.array(R2))
    T = np.array(T); print(tag, len(res), "residues,", len(T), "frames,", f"{T[0]:.0f}-{T[-1]:.0f} ps, dt {T[1]-T[0]:.0f} ps", flush=True)


def ris(Rm, vm, l2, n):
    I = np.eye(3); ev = np.linalg.eigvals(Rm); cp = ev[np.abs(ev.imag) > 1e-8]
    corr = lambda k: (vm @ np.linalg.matrix_power(Rm, k) @ vm) / l2
    re = ev[np.abs(ev.imag) <= 1e-8]
    return dict(Cinf=float(1 + 2 * (vm @ Rm @ np.linalg.inv(I - Rm) @ vm) / l2), c1=float(corr(1)), c2=float(corr(2)), c3=float(corr(3)),
                rho=float(re[0].real) if len(re) else None, r=float(abs(cp[0])) if len(cp) else None,
                Cn_pred=float(1 + 2.0 / n * sum((n - k) * corr(k) for k in range(1, n))),
                period_bonds=float(2 * np.pi / abs(np.angle(cp[0]))) if len(cp) else None)


def tau(x, dt):
    x = x - x.mean(); n = len(x); v = x @ x / n
    if v == 0: return 0.0
    s = 0.0
    for k in range(1, n // 2):
        c = (x[:-k] @ x[k:]) / (n - k) / v
        if c <= 0: break
        s += c
    return float(dt * (1 + 2 * s))


def one(tag):
    from ring_factor_blocks import rigid_RV
    d = np.load(REP / f"{tag}.npz"); m = d["t"] >= T0_PS - 1e-6
    t, R, v, pp, R2 = d["t"][m], d["R"][m].astype(float), d["v"][m].astype(float), d["phipsi"][m].astype(float), d["R2"][m]
    F, L = pp.shape[:2]; psiH = (pp[..., 1] + 117.8 + 180) % 360 - 180; anti = np.abs(psiH) > 90
    Rr, Vr = rigid_RV(pp.reshape(-1, 2)); Rr = Rr.reshape(F, L, 3, 3); Vr = Vr.reshape(F, L, 3)
    def block(s):
        l2 = np.mean(np.sum(v[s] ** 2, -1)); fl = ris(R[s].mean((0, 1)), v[s].mean((0, 1)), l2, L)
        rg = ris(Rr[s].mean((0, 1)), Vr[s].mean((0, 1)), np.mean(np.sum(Vr[s] ** 2, -1)), L)
        syn = ~anti[s]; a = np.radians(psiH[s][syn]); b = np.radians(pp[s][..., 0][syn])
        return dict(p_anti=float(anti[s].mean()), Cinf_flex=fl["Cinf"], Cinf_rigid=rg["Cinf"], ring_factor=fl["Cinf"] / rg["Cinf"],
                    c1=fl["c1"], c2=fl["c2"], c3=fl["c3"], rho=fl["rho"], r=fl["r"], c2_rigid=rg["c2"], period_rigid=rg["period_bonds"], period_bonds=fl["period_bonds"], Cn_pred=fl["Cn_pred"], Cn_meas=float(R2[s].mean() / (L * l2)), l_nm=float(np.sqrt(l2)),
                    psiH_syn=float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))), phi_syn=float(np.degrees(np.arctan2(np.sin(b).mean(), np.cos(b).mean()))))
    out = block(slice(None)); e = np.linspace(0, F, NBLOCK + 1).astype(int)
    bl = [block(slice(a, b)) for a, b in zip(e[:-1], e[1:])]
    out["block_se"] = {k: float(np.std([b[k] for b in bl], ddof=1) / np.sqrt(NBLOCK)) for k in ("p_anti", "Cinf_flex", "Cinf_rigid", "ring_factor", "c1", "c2", "rho", "r", "period_bonds", "Cn_pred", "Cn_meas")}
    out["p_anti_per_linkage"] = anti.mean(0).tolist()
    out.update(frames=int(F), linkages=int(L), t0_ps=float(t[0]), t1_ps=float(t[-1]), block_ns=float((t[-1] - t[0]) / NBLOCK / 1000),
               tau_flips_ps=tau(anti.sum(1).astype(float), float(t[1] - t[0])), tag=tag)
    return out


def stats(name, tags):
    from multiprocessing import Pool
    with Pool(min(len(tags), 5)) as p: rows = p.map(one, tags)
    keys = ("p_anti", "Cinf_flex", "Cinf_rigid", "ring_factor", "c1", "period_bonds", "Cn_pred", "Cn_meas", "psiH_syn", "phi_syn", "l_nm", "c2", "c3", "rho", "r", "c2_rigid", "period_rigid")
    print(f"{name}: t >= {T0_PS/1000:.0f} ns; block SE from {NBLOCK} blocks per trajectory")
    print(f"{'trajectory':16s} {'ns':>5s} {'p_anti %':>14s} {'Cinf flex':>12s} {'Cinf rigid':>12s} {'factor':>14s} {'c1':>6s} {'period':>6s} {'Cn pred/meas':>12s} {'psiH':>6s} {'tau_f ns':>8s}")
    for r in rows:
        s = r["block_se"]
        print(f"{r['tag']:16s} {r['t1_ps']/1000:5.0f} {100*r['p_anti']:6.2f} +- {100*s['p_anti']:4.2f} {r['Cinf_flex']:5.2f} +- {s['Cinf_flex']:4.2f} {r['Cinf_rigid']:5.2f} +- {s['Cinf_rigid']:4.2f}"
              f" {r['ring_factor']:6.3f} +- {s['ring_factor']:5.3f} {r['c1']:6.3f} {r['period_bonds']:6.2f} {r['Cn_pred']:5.2f}/{r['Cn_meas']:5.2f} {r['psiH_syn']:6.1f} {r['tau_flips_ps']/1000:8.2f}")
    out = dict(trajectories=rows)
    if len(rows) > 1:
        out["group"] = {k: dict(mean=float(np.mean([r[k] for r in rows])), se=float(np.std([r[k] for r in rows], ddof=1) / np.sqrt(len(rows))),
                                sd=float(np.std([r[k] for r in rows], ddof=1))) for k in keys}
        g = out["group"]; print(f"{'mean +- SE (n=' + str(len(rows)) + ')':22s}", "  ".join(f"{k} {g[k]['mean']:.3f} +- {g[k]['se']:.3f}" for k in keys[:8]))
    (RES / f"replica_{name}.json").write_text(json.dumps(out, indent=1))


def table():
    """LaTeX rows for the SI table of the long runs -> ../manuscript_JPCB/tab_replicas.tex"""
    name = lambda t: ("GLYCAM06 r" + t.split("_r")[1]) if t.startswith("GLY") else ("\\SC{}" + t.split("_")[0][2:] + (" r" + t.split("_r")[1] if "_r" in t and "run1" not in t else (", first run" if "run1" in t else ", extension")))
    f = lambda r, s=None: (f"{100*r['p_anti']:.1f} ({100*s['p_anti']:.1f}) & {r['Cinf_flex']:.2f} ({s['Cinf_flex']:.2f}) & {r['Cinf_rigid']:.2f} & {r['ring_factor']:.3f} ({s['ring_factor']:.3f}) & "
                           f"{r['c1']:.3f} & {r['c2']:.2f} & {r['period_bonds']:.2f} & {r['Cn_pred']:.2f} / {r['Cn_meas']:.2f} & ${r['psiH_syn']:.0f}$")
    L = []
    for key, lab in (("SC6_charmm_rep", "\\SC{}6, five replicas"), ("SC12_charmm_rep", "\\SC{}12, five replicas"), ("charmm_long", None), ("SC6_glycam", "GLYCAM06 \\SC{}6, five replicas")):
        d = json.loads((RES / f"replica_{key}.json").read_text())
        for r in d["trajectories"]: L.append(f"{name(r['tag'])} & {r['t1_ps']/1000:.0f} & " + f(r, r["block_se"]) + " \\\\")
        if not lab: L.append("\\midrule")
        if lab:
            g = d["group"]; m = {k: g[k]["mean"] for k in g}; e = {k: g[k]["se"] for k in g}
            L.append("\\textbf{" + lab + "} & & " + f(m, e) + " \\\\"); L.append("\\midrule")
    L = L[:-1] if L[-1] == "\\midrule" else L
    (HERE.parent / "manuscript_JPCB" / "tab_replicas.tex").write_text("\n".join(L) + "\n\\bottomrule\n"); print("\n".join(L))


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    if sys.argv[1] == "extract": extract(*sys.argv[2:5])
    elif sys.argv[1] == "table": table()
    else: stats(sys.argv[2], sys.argv[3:])
