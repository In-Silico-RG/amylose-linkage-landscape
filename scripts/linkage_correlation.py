#!/usr/bin/env python3
"""Test of the independent-linkage (RIS) assumption on the MD, and block errors on the
residue-frame transfer matrix (referee W1 and W3, 2026-09-22).

Per frame and strand: residue frames F_i (C1->C4, O5), transforms R_i = F_i^T F_{i+1}, the
virtual bond v_i in the frame of the residue it spans, and the linkage torsions
phi_i = O5(i+1)-C1(i+1)-O4(i)-C4(i), psi_i = C1(i+1)-O4(i)-C4(i)-C5(i).

Tests
  T1  joint band-flip statistics: P(anti_i & anti_{i+1}) vs P(anti)^2, P(anti_{i+1}|anti_i).
  T2  circular correlation (Jammalamadaka-Sarma) between neighbouring torsions.
  T3  pair independence of transforms: ||<R_i R_{i+1}> - <R>^2||_F / ||<R>^2||_F, and the
      2- and 3-bond correlations from the exact products vs from <R>^k.
  T4  conditional transfer matrices <R | previous linkage syn / anti> and their eigenvalues.
  T5  block errors (4 x 2 ns) on rho, r, period, C_inf of <R>.
Output: ../resultados/linkage_corr_<SYS>.json and a printed summary.
"""
import sys, json
from pathlib import Path
import numpy as np
import MDAnalysis as mda
from chain_stats import SYSTEMS, strands, T_START_PS, DT_PS, make_chain_whole
HERE = Path(__file__).resolve().parent


def frame_of(C1, C4, O5):
    e1 = C4 - C1; e1 /= np.linalg.norm(e1)
    u = O5 - C1; e2 = u - np.dot(u, e1) * e1; e2 /= np.linalg.norm(e2)
    return np.column_stack([e1, e2, np.cross(e1, e2)])


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n; w = b2 - np.dot(b2, b1n) * b1n
    return np.degrees(np.arctan2(np.dot(np.cross(b1n, v), w), np.dot(v, w)))


def circ_corr(a, b):
    a, b = np.radians(a), np.radians(b)
    am = np.arctan2(np.sin(a).mean(), np.cos(a).mean()); bm = np.arctan2(np.sin(b).mean(), np.cos(b).mean())
    return float(np.sum(np.sin(a - am) * np.sin(b - bm)) / np.sqrt(np.sum(np.sin(a - am) ** 2) * np.sum(np.sin(b - bm) ** 2)))


def eig_summary(Rm, vm, l2, n):
    I = np.eye(3); ev = np.linalg.eigvals(Rm)
    re = ev[np.abs(ev.imag) <= 1e-8]; cp = ev[np.abs(ev.imag) > 1e-8]
    corr = lambda k: (vm @ np.linalg.matrix_power(Rm, k) @ vm) / l2
    return dict(rho=float(re.real.max()) if len(re) else None, r=float(abs(cp[0])) if len(cp) else None,
                period=float(2 * np.pi / abs(np.angle(cp[0]))) if len(cp) else None,
                Cinf=float(1 + 2 * (vm @ Rm @ np.linalg.inv(I - Rm) @ vm) / l2),
                Cn=float(1 + 2.0 / n * sum((n - k) * corr(k) for k in range(1, n))))


def analyse(system):
    top, trj = SYSTEMS[system]
    u = mda.Universe(str(top), str(trj)); chains = strands(u)
    stride = max(1, int(round(DT_PS / u.trajectory.dt)))
    N = len(chains[0]); n = N - 1
    rec = []   # per frame: dict(R (n,3,3), v (n,3), phi (n,), psi (n,), t)
    for ts in u.trajectory[::stride]:
        if ts.time < T_START_PS - 1e-6: continue
        box = ts.dimensions[:3]
        for c in chains:
            raw = np.array([r.atoms.select_atoms("name O4").positions[0] for r in c])
            O4 = make_chain_whole(raw, box)
            P = []
            for r, o_raw, o in zip(c, raw, O4):
                p = {nm: r.atoms.select_atoms("name " + nm).positions[0] for nm in ("C1", "C4", "C5", "O5")}
                for nm in p:
                    d = p[nm] - o_raw; d -= box * np.round(d / box); p[nm] = o + d
                p["O4"] = o; P.append(p)
            F = [frame_of(p["C1"], p["C4"], p["O5"]) for p in P]
            R = np.array([F[i].T @ F[i + 1] for i in range(n)])
            v = np.array([F[i + 1].T @ (O4[i + 1] - O4[i]) / 10.0 for i in range(n)])
            phi = np.array([dihedral(P[i + 1]["O5"], P[i + 1]["C1"], P[i]["O4"], P[i]["C4"]) for i in range(n)])
            psi = np.array([dihedral(P[i + 1]["C1"], P[i]["O4"], P[i]["C4"], P[i]["C5"]) for i in range(n)])
            rec.append(dict(R=R, v=v, phi=phi, psi=psi, t=ts.time))
    R = np.array([r["R"] for r in rec]); V = np.array([r["v"] for r in rec])
    PHI = np.array([r["phi"] for r in rec]); PSI = np.array([r["psi"] for r in rec]); T = np.array([r["t"] for r in rec])
    Rm = R.mean((0, 1)); vm = V.mean((0, 1)); l2 = float(np.mean(np.sum(V ** 2, axis=2)))
    out = dict(system=system, N=N, n=n, frames=len(rec) // len(chains), strands=len(chains))
    # T1 band flips
    psiH = (PSI + 117.8 + 180) % 360 - 180; anti = np.abs(psiH) > 90
    pa = anti.mean(); pj = (anti[:, :-1] & anti[:, 1:]).mean()
    out["T1"] = dict(p_anti=float(pa), p_joint=float(pj), p_indep=float(pa ** 2),
                     p_anti_next_given_anti=float(pj / pa) if pa > 0 else None,
                     n_anti_per_chain=float(anti.sum(1).mean()))
    # T2 circular correlations between neighbours (pooled over linkages and frames)
    a = lambda X: X[:, :-1].ravel(); b = lambda X: X[:, 1:].ravel()
    out["T2"] = dict(psi_psi=circ_corr(a(PSI), b(PSI)), phi_phi=circ_corr(a(PHI), b(PHI)),
                     phi_psi_next=circ_corr(a(PHI), b(PSI)), psi_phi_next=circ_corr(a(PSI), b(PHI)),
                     psi_psi_lag2=circ_corr(PSI[:, :-2].ravel(), PSI[:, 2:].ravel()) if n > 2 else None)
    # T3 pair independence of transforms
    RR = np.einsum("fiab,fibc->fiac", R[:, :-1], R[:, 1:]).mean((0, 1)); R2 = Rm @ Rm
    c2_exact = float(vm @ RR @ vm / l2); c2_ris = float(vm @ R2 @ vm / l2)
    c2_meas = float(np.mean(np.sum(V[:, :-1] * np.einsum("fiab,fib->fia", R[:, :-1], V[:, 1:]), axis=2)) / l2)
    res = dict(frob_rel=float(np.linalg.norm(RR - R2) / np.linalg.norm(R2)), c2_from_pair_products=c2_exact,
               c2_from_Rmean2=c2_ris, c2_measured=c2_meas)
    if n > 2:
        RRR = np.einsum("fiab,fibc,ficd->fiad", R[:, :-2], R[:, 1:-1], R[:, 2:]).mean((0, 1))
        res["c3_from_triple_products"] = float(vm @ RRR @ vm / l2); res["c3_from_Rmean3"] = float(vm @ Rm @ Rm @ Rm @ vm / l2)
    out["T3"] = res
    # T4 conditional on previous linkage state
    prev_anti = anti[:, :-1]; Rn = R[:, 1:]
    cond = {}
    for lab, m in (("prev_syn", ~prev_anti), ("prev_anti", prev_anti)):
        if m.sum() > 50:
            Rc = Rn[m].mean(0); cond[lab] = dict(count=int(m.sum()), **eig_summary(Rc, vm, l2, n))
    out["T4"] = cond
    # T5 block errors
    blocks = np.array_split(np.arange(len(rec)), 4); vals = []
    for bl in blocks:
        vals.append(eig_summary(R[bl].mean((0, 1)), V[bl].mean((0, 1)), float(np.mean(np.sum(V[bl] ** 2, axis=2))), n))
    keys = ["rho", "r", "period", "Cinf", "Cn"]
    out["T5"] = {k: dict(mean=float(np.mean([v[k] for v in vals])), se=float(np.std([v[k] for v in vals], ddof=1) / 2)) for k in keys if all(v[k] is not None for v in vals)}
    out["full"] = eig_summary(Rm, vm, l2, n)
    t1, t2, t3, t5, f = out["T1"], out["T2"], out["T3"], out["T5"], out["full"]
    print(f"{system:5s} N={N:2d} | T1 anti {100*t1['p_anti']:.1f}% joint {100*t1['p_joint']:.2f}% indep {100*t1['p_indep']:.2f}% "
          f"P(anti|anti) {t1['p_anti_next_given_anti'] if t1['p_anti_next_given_anti'] is None else round(t1['p_anti_next_given_anti'],2)} "
          f"| T2 psi-psi {t2['psi_psi']:+.2f} phi-phi {t2['phi_phi']:+.2f} phi-psi' {t2['phi_psi_next']:+.2f} psi-phi' {t2['psi_phi_next']:+.2f} "
          f"| T3 frob {t3['frob_rel']:.3f} c2 pair {t3['c2_from_pair_products']:.3f} vs <R>^2 {t3['c2_from_Rmean2']:.3f} (meas {t3['c2_measured']:.3f})"
          + (f" c3 {t3['c3_from_triple_products']:.3f} vs {t3['c3_from_Rmean3']:.3f}" if 'c3_from_triple_products' in t3 else "")
          + f" | T5 rho {t5['rho']['mean']:.3f}±{t5['rho']['se']:.3f} r {t5['r']['mean']:.3f}±{t5['r']['se']:.3f} P {t5['period']['mean']:.2f}±{t5['period']['se']:.2f} Cinf {t5['Cinf']['mean']:.2f}±{t5['Cinf']['se']:.2f}"
          + " | T4 " + " ".join(f"{k}: rho {v['rho']:.3f} r {v['r']:.3f} P {v['period']:.2f} Cinf {v['Cinf']:.1f} (n={v['count']})" for k, v in cond.items()), flush=True)
    (HERE.parent / "resultados" / f"linkage_corr_{system}.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    for s in sys.argv[1:]: analyse(s)
