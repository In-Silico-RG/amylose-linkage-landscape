#!/usr/bin/env python3
"""Ring-flexibility factor C_inf(flexible residues) / C_inf(rigid template, same torsions), with block errors
(reviewers r7: Kimi W4, DeepSeek W4 — the 0.73 of SI Table S9 had no uncertainty). AFC 2026-09-25 "go".

Stage A (needs MDAnalysis: ~/miniconda3/bin/python):  extract SYS ...
    per analysed frame and linkage: flexible residue-frame rotation R_i = F_i^T F_{i+1}, virtual bond
    v_i = F_{i+1}^T (O4_{i+1} - O4_i) (conventions of residue_frame_ris.py), and the linkage torsions
    phi = O5(i+1)-C1(i+1)-O4(i)-C4(i), psi = C1(i+1)-O4(i)-C4(i)-C5(i)  -> ../resultados/ringfac_<SYS>.npz
Stage B (needs RDKit, chem env):  blocks SYS ...
    the same torsions placed on the rigid xtb template (cheminf/chain_builder.py), residue frames from
    C1, C4, O5 exactly as for the trajectories; exact C_inf of both per 2-ns block (4 blocks after 2 ns)
    and pooled; ratio per block -> mean +- SE     -> ../resultados/ring_factor_blocks.json
"""
import sys, json
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
RES = HERE.parent / "resultados"
TEMPLATE = "maltose_a_84_-145.xyz"      # the template of mc_chains.py / the paper (map C_inf 7.33)


def cinf(Rm, vm, l2):
    I = np.eye(3)
    return float(1 + 2 * (vm @ Rm @ np.linalg.inv(I - Rm) @ vm) / l2)


def frame_of(C1, C4, O5):                      # same as residue_frame_ris.frame_of (no MDAnalysis import)
    e1 = C4 - C1; e1 /= np.linalg.norm(e1)
    u = O5 - C1; e2 = u - np.dot(u, e1) * e1; e2 /= np.linalg.norm(e2)
    return np.column_stack([e1, e2, np.cross(e1, e2)])


def dih(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1); v = b0 - b0 @ b1 * b1; w = b2 - b2 @ b1 * b1
    return np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w))


def extract(system):
    import MDAnalysis as mda
    sys.path.insert(0, str(HERE))
    from chain_stats import SYSTEMS, strands, T_START_PS, DT_PS, make_chain_whole
    from residue_frame_ris import frame_of
    top, trj = SYSTEMS[system]
    u = mda.Universe(str(top), str(trj)); chains = strands(u)
    stride = max(1, int(round(DT_PS / u.trajectory.dt)))
    T, R, V, PP = [], [], [], []
    for ts in u.trajectory[::stride]:
        if ts.time < T_START_PS - 1e-6: continue
        box = ts.dimensions[:3]
        for c in chains:
            raw = np.array([r.atoms.select_atoms("name O4").positions[0] for r in c])
            O4 = make_chain_whole(raw, box); F, P = [], []
            for r, o_raw, o in zip(c, raw, O4):
                p = {nm: r.atoms.select_atoms("name " + nm).positions[0] for nm in ("C1", "C4", "O5", "C5", "O4")}
                for nm in p:
                    d = p[nm] - o_raw; d -= box * np.round(d / box); p[nm] = o + d
                F.append(frame_of(p["C1"], p["C4"], p["O5"])); P.append(p)
            for i in range(len(c) - 1):
                a, b = P[i], P[i + 1]
                T.append(ts.time); R.append(F[i].T @ F[i + 1]); V.append(F[i + 1].T @ (O4[i + 1] - O4[i]) / 10.0)
                PP.append((dih(b["O5"], b["C1"], a["O4"], a["C4"]), dih(b["C1"], a["O4"], a["C4"], a["C5"])))
    np.savez(RES / f"ringfac_{system}.npz", t=np.array(T), R=np.array(R), v=np.array(V), phipsi=np.array(PP))
    print(system, len(T), "linkage samples")


def rigid_RV(phipsi):
    sys.path.insert(0, str(HERE.parent / "cheminf"))
    from chain_builder import Template, build_chain
    Tp = Template(HERE.parent / "cheminf" / TEMPLATE)
    idx = {a: k for k, a in enumerate(Tp.res)}
    fr = lambda r: frame_of(r[idx[Tp.C1p]], r[idx[Tp.C4p]], r[idx[Tp.O5p]])
    R, V = [], []
    # The builder runs toward the reducing end (residue 0's C1 carries the glycosidic O bonded to
    # residue 1's C4); the MD analysis runs the other way (C1 of i+1 bonded to O4 of i). In MD order
    # residue i = builder residue 1, residue i+1 = builder residue 0, so R_md = F1^T F0 and the bond
    # spanning residue i+1 (= builder residue 0) is O4_{i+1} - O4_i = O4s[0] - O4s[1] in frame F0.
    for ph, ps in phipsi:
        res, O4 = build_chain(Tp, [(ph, ps)])
        F0, F1 = fr(res[0]), fr(res[1])
        R.append(F1.T @ F0); V.append(F0.T @ (O4[0] - O4[1]) / 10.0)
    return np.array(R), np.array(V)


def blocks(systems):
    out = {}
    for s in systems:
        d = np.load(RES / f"ringfac_{s}.npz"); t = d["t"]
        Rr, Vr = rigid_RV(d["phipsi"])
        edges = np.linspace(t.min(), t.max() + 1e-6, 5)
        rows = []
        for lo, hi in list(zip(edges[:-1], edges[1:])) + [(t.min(), t.max() + 1e-6)]:
            m = (t >= lo) & (t < hi)
            cf = cinf(d["R"][m].mean(0), d["v"][m].mean(0), np.mean(np.sum(d["v"][m] ** 2, 1)))
            cr = cinf(Rr[m].mean(0), Vr[m].mean(0), np.mean(np.sum(Vr[m] ** 2, 1)))
            rows.append(dict(t0_ps=float(lo), t1_ps=float(hi), n=int(m.sum()), Cinf_flex=cf, Cinf_rigid=cr, ratio=cf / cr))
        rb = np.array([r["ratio"] for r in rows[:4]])
        out[s] = dict(blocks=rows[:4], pooled=rows[4], ratio_mean=float(rb.mean()), ratio_se=float(rb.std(ddof=1) / 2))
        print(s, " ".join(f"{r['Cinf_flex']:.2f}/{r['Cinf_rigid']:.2f}={r['ratio']:.3f}" for r in rows),
              f"| blocks {rb.mean():.3f} +- {rb.std(ddof=1) / 2:.3f}")
    allr = np.array([r["ratio"] for s in out for r in out[s]["blocks"]])
    out["all_blocks"] = dict(n=len(allr), mean=float(allr.mean()), se=float(allr.std(ddof=1) / np.sqrt(len(allr))), sd=float(allr.std(ddof=1)))
    print("all blocks:", out["all_blocks"])
    (RES / "ring_factor_blocks.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"extract": lambda a: [extract(s) for s in a], "blocks": blocks}[sys.argv[1]](sys.argv[2:])
