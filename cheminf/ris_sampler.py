#!/usr/bin/env python3
"""RIS / Monte Carlo chain statistics from a (phi, psi) distribution (WP6, WP6b).

Sources of the linkage distribution:
  --from-md SYS      empirical (phi, psi) pairs measured in the MD (phipsi_SYS.npy);
                     tests the rigid-residue + independent-linkage approximations alone.
  --from-map FILE    Boltzmann weights exp(-E/kT) on the xtb (phi, psi) grid (scan_10deg.dat).
Linkages are drawn independently (RIS assumption). Chains of N residues are built with
chain_builder.build_chain; per chain we get l, tangent correlation, C_n and the local-frame
transfer matrix; averages over --nchains chains. Optional hard-sphere excluded volume between
O4 nodes farther than 2 apart (--ev radius_nm) by rejection.

Output: ../resultados/ris_<source>.json and a printed summary comparable to chain_stats.py.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from chain_builder import Template, build_chain, chain_stats_from_nodes, HERE

KT_KCAL = 0.0019872 * 303.15


def load_md(sysn):
    pp = np.load(HERE / f"phipsi_{sysn}.npy")
    return pp, np.ones(len(pp)) / len(pp)


def load_map(f):
    rows = [l.split() for l in open(f) if len(l.split()) >= 3 and l.split()[0] != "DONE"]
    if rows and len(rows[0]) >= 6:                   # validated scan (validate_frames.py): keep valid == 1
        nbad = sum(1 for r in rows if r[5] != "1")
        print(f"validated scan: {nbad} of {len(rows)} frames excluded (connectivity / ring / drift)")
        rows = [r for r in rows if r[5] == "1"]
    a = np.array([[float(x) for x in r[:5]] for r in rows if len(r) >= 5]) if all(len(r) >= 5 for r in rows) \
        else np.array([[float(x) for x in r[:3]] for r in rows])
    if a.shape[1] == 5:                             # drop points whose final torsion drifted > 5 deg
        dphi = (a[:, 3] - a[:, 0] + 180) % 360 - 180; dpsi = (a[:, 4] - a[:, 1] + 180) % 360 - 180
        ok = (np.abs(dphi) < 5) & (np.abs(dpsi) < 5)
        if (~ok).any(): print(f"load_map: {(~ok).sum()} points dropped (torsion drift > 5 deg)")
        a = a[ok]
    E = (a[:, 2] - a[:, 2].min()) * 627.509        # Eh -> kcal/mol, relative
    w = np.exp(-E / KT_KCAL); w /= w.sum()
    return a[:, :2], w, E


def ris_T_stats(Tm, l, n):
    ev = np.linalg.eigvals(Tm)
    corr = [np.linalg.matrix_power(Tm, k)[0, 0] for k in range(n)]
    Cn = 1 + 2.0 / n * sum((n - k) * corr[k] for k in range(1, n))
    I = np.eye(3)
    Cinf = ((I + Tm) @ np.linalg.inv(I - Tm))[0, 0]
    cp = ev[np.abs(ev.imag) > 1e-8]
    period = 2 * np.pi / abs(np.angle(cp[0])) if len(cp) else None
    return dict(Cn=float(Cn), Cinf=float(Cinf), lam_max=float(max(abs(ev))),
                period_bonds=period, period_nm=None if period is None else period * l,
                eig=[[float(e.real), float(e.imag)] for e in ev], corr=[float(c) for c in corr])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-md"); ap.add_argument("--from-map")
    ap.add_argument("--N", type=int, nargs="+", default=[6, 12, 18, 24, 40, 100])
    ap.add_argument("--nchains", type=int, default=2000)
    ap.add_argument("--ev", type=float, default=0.0, help="hard-sphere radius on O4 nodes, nm (0 = none)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--template", default=str(HERE / "maltose_a_84_-145.xyz"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    if a.from_md:
        pp, w = load_md(a.from_md); src = f"md_{a.from_md}"; E = None
    else:
        pp, w, E = load_map(a.from_map); src = "map_" + Path(a.from_map).stem
    T = Template(a.template)
    out = dict(source=src, nchains=a.nchains, ev_nm=a.ev, results={})
    Tsum = np.zeros((3, 3)); Tcnt = 0
    for N in a.N:
        n = N - 1; stats = []; rejected = 0
        while len(stats) < a.nchains:
            idx = rng.choice(len(pp), size=n, p=w)
            pairs = pp[idx] + (rng.uniform(-5, 5, size=(n, 2)) if E is not None else 0)  # jitter within grid cell
            res, O4 = build_chain(T, pairs)
            x = O4 / 10.0
            if a.ev > 0:
                D = np.linalg.norm(x[:, None] - x[None], axis=2)
                iu = np.triu_indices(N, 3)
                if np.any(D[iu] < 2 * a.ev):
                    rejected += 1; continue
            stats.append(chain_stats_from_nodes(x))
        l = np.mean([s["l"] for s in stats])
        corr = np.mean([s["corr"] for s in stats], axis=0)
        Cn = np.mean([s["Cn"] for s in stats])
        R2 = Cn * n * l ** 2
        Tm = np.mean([s["T"] for s in stats], axis=0)
        if N == max(a.N):
            Tsum = Tm
        ris = ris_T_stats(Tm, l, n)
        out["results"][N] = dict(l=float(l), Cn=float(Cn), Cn_se=float(np.std([s["Cn"] for s in stats]) / np.sqrt(len(stats))),
                                 corr=corr.tolist(), T=Tm.tolist(), RIS=ris, rejected=rejected)
        print(f"{src} N={N:3d}: l={l:.3f} nm  C_n={Cn:.2f}±{out['results'][N]['Cn_se']:.2f}  RIS(<T>) C_n={ris['Cn']:.2f} "
              f"C_inf={ris['Cinf']:.2f} |lam|={ris['lam_max']:.3f} period={ris['period_bonds'] or float('nan'):.2f} bonds "
              f"({(ris['period_nm'] or float('nan')):.2f} nm)  corr k=1..5: {np.round(corr[1:6], 2)}  rejected={rejected}")
    if E is not None:
        out["map"] = dict(Emin_kcal=0.0, grid_points=int(len(E)), w_max=float(w.max()),
                          phipsi_mean=[float(np.degrees(np.angle(np.sum(w * np.exp(1j * np.radians(pp[:, i])))))) for i in (0, 1)])
    dest = HERE.parent / "resultados" / f"ris_{src}{'_ev' if a.ev else ''}.json"
    dest.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
