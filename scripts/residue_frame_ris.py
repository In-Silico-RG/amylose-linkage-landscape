#!/usr/bin/env python3
"""Residue-frame transfer matrix <R> from the MD (D-F8 check, 2026-09-22).

For every linkage i -> i+1 and every analysed frame: residue frame F_i from the ring atoms
(e1 = C1->C4, e2 = component of C1->O5 normal to e1, e3 = e1 x e2), rotation R_i = F_i^T F_{i+1},
virtual bond in the frame of the residue it spans, v_i = F_{i+1}^T (O4_{i+1} - O4_i)
(residue i's O4 sits on its C4 and is bonded to C1 of residue i+1). Averages give Flory's RIS with
independent linkages: corr(k) = <v>^T <R>^k <v> / <|v|^2>, C_inf = 1 + 2 <v>^T <R> (I - <R>)^-1 <v> / <|v|^2>.
Compared with the bond-frame <T> (chain_stats.py) and the measured C_n of the same frames.

    ~/miniconda3/bin/python residue_frame_ris.py SC6 SC12 SC18 SC24 SC40 DC6
Output: ../resultados/ris_residue_frame.json
"""
import sys, json
import numpy as np
import MDAnalysis as mda
from pathlib import Path
from chain_stats import SYSTEMS, strands, T_START_PS, DT_PS, make_chain_whole
HERE = Path(__file__).resolve().parent


def frame_of(C1, C4, O5):
    e1 = C4 - C1; e1 /= np.linalg.norm(e1)
    u = O5 - C1; e2 = u - np.dot(u, e1) * e1; e2 /= np.linalg.norm(e2)
    return np.column_stack([e1, e2, np.cross(e1, e2)])


def bond_frames(bh):
    F = []
    for i in range(1, len(bh)):
        e1 = bh[i]; e2 = bh[i - 1] - np.dot(bh[i - 1], e1) * e1; e2 /= np.linalg.norm(e2)
        F.append(np.column_stack([e1, e2, np.cross(e1, e2)]))
    return F


def analyse(system):
    top, trj = SYSTEMS[system]
    u = mda.Universe(str(top), str(trj))
    chains = strands(u)
    stride = max(1, int(round(DT_PS / u.trajectory.dt)))
    Rs, vs, Ts, R2s, nfr = [], [], [], [], 0
    N = len(chains[0]); n = N - 1
    for ts in u.trajectory[::stride]:
        if ts.time < T_START_PS - 1e-6:
            continue
        nfr += 1; box = ts.dimensions[:3]
        for c in chains:
            raw = np.array([r.atoms.select_atoms("name O4").positions[0] for r in c])
            O4 = make_chain_whole(raw, box)                       # unwrapped nodes, Angstrom
            F = []
            for r, o_raw, o in zip(c, raw, O4):
                p = {nm: r.atoms.select_atoms("name " + nm).positions[0] for nm in ("C1", "C4", "O5")}
                for nm in p:                                      # ring whole around O4, then shifted with the node
                    d = p[nm] - o_raw; d -= box * np.round(d / box); p[nm] = o + d
                F.append(frame_of(p["C1"], p["C4"], p["O5"]))
            for i in range(n):
                Rs.append(F[i].T @ F[i + 1])
                vs.append(F[i + 1].T @ (O4[i + 1] - O4[i]) / 10.0)   # O4_i is on C4 of residue i and bonded to C1 of i+1: the bond O4_i -> O4_{i+1} spans residue i+1
            b = np.diff(O4, axis=0) / 10.0; bh = b / np.linalg.norm(b, axis=1)[:, None]
            Fb = bond_frames(bh); Ts += [Fb[j].T @ Fb[j + 1] for j in range(len(Fb) - 1)]
            R2s.append(np.sum((O4[-1] - O4[0]) ** 2) / 100.0)
    Rm = np.mean(Rs, 0); vm = np.mean(vs, 0); l2 = np.mean([v @ v for v in vs]); Tm = np.mean(Ts, 0)
    I = np.eye(3)
    def summary(Mtx, kind):
        ev = np.linalg.eigvals(Mtx)
        if kind == "residue":
            corr = lambda k: (vm @ np.linalg.matrix_power(Mtx, k) @ vm) / l2
            Cinf = 1 + 2 * (vm @ Mtx @ np.linalg.inv(I - Mtx) @ vm) / l2
        else:
            corr = lambda k: np.linalg.matrix_power(Mtx, k)[0, 0]
            Cinf = ((I + Mtx) @ np.linalg.inv(I - Mtx))[0, 0]
        cp = ev[np.abs(ev.imag) > 1e-8]; re = ev[np.abs(ev.imag) <= 1e-8]
        Cn = 1 + 2.0 / n * sum((n - k) * corr(k) for k in range(1, n))
        return dict(Cinf=float(Cinf), Cn_pred=float(Cn), lam_real=[float(x.real) for x in re],
                    lam_helix=float(abs(cp[0])) if len(cp) else None,
                    period_bonds=float(2 * np.pi / abs(np.angle(cp[0]))) if len(cp) else None,
                    corr=[float(corr(k)) for k in range(6)])
    res = dict(system=system, N=N, n=n, frames=nfr, strands=len(chains), l=float(np.sqrt(l2)),
               v_mean_frame=vm.tolist(), v_mean_norm=float(np.linalg.norm(vm)),
               Cn_measured=float(np.mean(R2s) / (n * l2)), R_mean=Rm.tolist(), T_mean=Tm.tolist(), l2=float(l2),
               residue_frame=summary(Rm, "residue"), bond_frame=summary(Tm, "bond"))
    r, b = res["residue_frame"], res["bond_frame"]
    print(f"{system:6s} N={N:2d} frames={nfr}  C_n meas={res['Cn_measured']:.2f} | residue frame: C_n pred={r['Cn_pred']:.2f} "
          f"C_inf={r['Cinf']:.2f} axis={r['lam_real']} helix|λ|={r['lam_helix']:.3f} period={r['period_bonds']:.2f} corr1-3={np.round(r['corr'][1:4],2)}"
          f" | bond frame: C_n pred={b['Cn_pred']:.2f} C_inf={b['Cinf']:.2f} axis={b['lam_real']} helix|λ|={b['lam_helix']:.3f} period={b['period_bonds']:.2f}"
          f" | |<v>|/l={res['v_mean_norm']/res['l']:.3f}", flush=True)
    return res


if __name__ == "__main__":
    out = {s: analyse(s) for s in sys.argv[1:]}
    (HERE.parent / "resultados" / "ris_residue_frame.json").write_text(json.dumps(out, indent=1))
