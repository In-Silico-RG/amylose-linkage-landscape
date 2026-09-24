#!/usr/bin/env python3
"""WP6b: long rigid-residue amylose chains (DP up to 1000) from a (phi, psi) map, with optional
hard-sphere excluded volume by Rosenbluth-weighted growth.

Each valid map cell is turned into a rigid transform (residue i frame -> residue i+1 frame),
computed once on the template with chain_builder's nerf + Kabsch placement, so a chain is a
product of 4x4 matrices. Cells are drawn with Boltzmann weight exp(-E/kT) (T = 303.15 K); a
+-5 deg jitter inside the cell is represented by NJIT precomputed variants per cell.

Excluded volume: hard spheres of radius --ev (nm) on the O4 nodes, |i - j| >= 3. Growth with
k trial cells per step; the chain weight is prod(m_i / k) (Rosenbluth); averages are
weight-weighted and the effective sample size ESS = (sum W)^2 / sum W^2 is reported.

Per N: C_n = <R^2>/(n l^2), <Rg^2>, nu from successive N, tangent correlation C(k) (k <= KMAX),
Kratky-Porod lp from a nonlinear exponential fit over windows [0, S] (theory R2 test).

    python3 mc_chains.py --from-map scan2_10deg.dat [--ev 0.25] [--N 6 ... 1000] [--nchains 400]
Output: ../resultados/mc_<map>[_ev<r>].json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from chain_builder import Template, nerf, kabsch, HERE
from ris_sampler import load_map, ris_T_stats

NJIT = 4
KMAX = 200
WINDOWS = (5, 11, 17, 23, 39, 99, 199)


def step_transform(T, phi, psi):
    """4x4 rigid transform taking template coordinates to the next residue, for one linkage."""
    X = T.X
    O5, C1, Og = X[T.O5p], X[T.C1p], X[T.Og]
    C4n = nerf(O5, C1, Og, T.b_OC4, T.a_C1OC4, phi)
    C5n = nerf(C1, Og, C4n, T.b_C4C5, T.a_OC4C5, psi)
    R, t = kabsch(X[T.triad], np.array([Og, C4n, C5n]))
    A = np.eye(4); A[:3, :3] = R; A[:3, 3] = t
    return A


def precompute(T, pp, rng):
    """(ncell, NJIT+1, 4, 4): centre + NJIT jittered transforms per cell."""
    M = np.empty((len(pp), NJIT + 1, 4, 4))
    for c, (phi, psi) in enumerate(pp):
        M[c, 0] = step_transform(T, np.radians(phi), np.radians(psi))
        for j in range(1, NJIT + 1):
            d = rng.uniform(-5, 5, 2)
            M[c, j] = step_transform(T, np.radians(phi + d[0]), np.radians(psi + d[1]))
    return M


def grow(N, M, w, rng, ev, k, node_local, og_local):
    """One chain of N residues. Returns (nodes (N x 3, nm), weight)."""
    A = np.eye(4); nodes = np.empty((N, 3)); nodes[0] = node_local[:3]
    W = 1.0; cells = rng.choice(len(w), size=(N - 1, k), p=w); jit = rng.integers(0, NJIT + 1, size=(N - 1, k))
    for i in range(N - 1):
        if ev <= 0:
            A = A @ M[cells[i, 0], jit[i, 0]]
            nodes[i + 1] = (A @ node_local)[:3]
            continue
        cand = np.array([A @ M[cells[i, j], jit[i, j]] for j in range(k)])   # k x 4 x 4
        pos = cand[:, :3, :3] @ node_local[:3] + cand[:, :3, 3]               # k x 3
        if i + 1 >= 3:
            D = np.linalg.norm(pos[:, None, :] - nodes[None, :i - 1, :], axis=2)  # vs nodes 0..i-2
            ok = np.all(D >= 2 * ev, axis=1)
        else:
            ok = np.ones(k, bool)
        m = ok.sum()
        if m == 0:
            return None, 0.0
        W *= m / k
        j = rng.choice(np.flatnonzero(ok))
        A = cand[j]; nodes[i + 1] = pos[j]
    return nodes, W


def chain_obs(x):
    b = np.diff(x, axis=0); bl = np.linalg.norm(b, axis=1); bh = b / bl[:, None]; n = len(b)
    kmax = min(KMAX, n - 1)
    corr = np.array([np.mean(np.sum(bh[:n - kk] * bh[kk:], axis=1)) for kk in range(kmax + 1)])
    R2 = np.sum((x[-1] - x[0]) ** 2); Rg2 = np.mean(np.sum((x - x.mean(0)) ** 2, axis=1))
    return bl.mean(), R2, Rg2, corr


def kp_fit(corr, l, S):
    """Nonlinear least-squares fit of exp(-k l / lp) to corr[k], k = 0..S (all-k policy)."""
    S = min(S, len(corr) - 1); k = np.arange(S + 1); c = corr[:S + 1]
    from scipy.optimize import minimize_scalar
    f = lambda lp: np.sum((np.exp(-k * l / lp) - c) ** 2)
    r = minimize_scalar(f, bounds=(0.01, 200), method="bounded")
    return float(r.x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-map", required=True)
    ap.add_argument("--N", type=int, nargs="+", default=[6, 12, 18, 24, 40, 100, 200, 400, 1000])
    ap.add_argument("--nchains", type=int, default=400)
    ap.add_argument("--ev", type=float, default=0.0)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--template", default=str(HERE / "maltose_a_84_-145.xyz"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    pp, w, E = load_map(a.from_map)
    T = Template(a.template)
    M = precompute(T, pp, rng)
    node_local = np.r_[T.X[T.O4p], 1.0] / np.r_[10, 10, 10, 1]   # nm; transforms are in A -> scale below
    # transforms were built in Angstrom: convert translation to nm
    M[:, :, :3, 3] /= 10.0
    node_local = np.r_[T.X[T.O4p] / 10.0, 1.0]
    src = Path(a.from_map).stem + a.tag
    out = dict(source=src, nchains=a.nchains, ev_nm=a.ev, k=a.k, results={})
    prev = None
    for N in a.N:
        t0 = time.time(); n = N - 1; obs = []; Ws = []; dead = 0
        while len(obs) < a.nchains:
            x, W = grow(N, M, w, rng, a.ev, a.k, node_local, None)
            if x is None:
                dead += 1; continue
            obs.append(chain_obs(x)); Ws.append(W)
        Ws = np.array(Ws); Wn = Ws / Ws.sum(); ess = 1.0 / np.sum(Wn ** 2)
        l = np.sum(Wn * [o[0] for o in obs]); R2 = np.sum(Wn * [o[1] for o in obs]); Rg2 = np.sum(Wn * [o[2] for o in obs])
        kmax = min(len(o[3]) for o in obs); corr = np.sum(Wn[:, None] * np.array([o[3][:kmax] for o in obs]), axis=0)
        Cn = R2 / (n * l ** 2)
        R2_se = np.sqrt(np.sum(Wn ** 2 * (np.array([o[1] for o in obs]) - R2) ** 2))
        lps = {S: kp_fit(corr, l, S) for S in WINDOWS if S <= n - 1}
        nu = None
        if prev is not None:
            nu = 0.5 * np.log(Rg2 / prev[1]) / np.log(N / prev[0])
        prev = (N, Rg2)
        out["results"][N] = dict(l=float(l), Cn=float(Cn), Cn_se=float(R2_se / (n * l ** 2)), Rg2=float(Rg2), R2=float(R2),
                                 nu_local=None if nu is None else float(nu), ess=float(ess), dead=dead,
                                 corr=corr.tolist(), lp_KP=lps)
        print(f"{src} ev={a.ev} N={N:4d}: l={l:.3f} C_n={Cn:.2f}±{out['results'][N]['Cn_se']:.2f}  Rg={np.sqrt(Rg2):.2f} nm  "
              f"R2/Rg2={R2 / Rg2:.2f}  nu={'-' if nu is None else f'{nu:.3f}'}  ESS={ess:.0f}/{len(obs)} dead={dead}  "
              f"lp_KP(S)=" + " ".join(f"{S}:{v:.2f}" for S, v in lps.items()) + f"  [{time.time() - t0:.0f} s]", flush=True)
    dest = HERE.parent / "resultados" / f"mc_{src}{f'_ev{a.ev:g}' if a.ev else ''}.json"
    dest.write_text(json.dumps(out, indent=1)); print("wrote", dest)


if __name__ == "__main__":
    main()


def ris_exact(M, w, T, nlist=(6, 12, 18, 24, 40, 100, 1000)):
    """Exact RIS for independent linkages from the residue-frame mean transform <R> (3x3 rotation
    part of the precomputed step transforms, Boltzmann-weighted, jitter variants averaged).
    Bond i (O4_i -> O4_{i+1}) in residue-i frame is v = O4' -> Og of the template, identical for
    every residue, so <b_i . b_j> = v^T <R>^{|j-i|} v exactly (Flory). Returns C_n(N), C_inf,
    eigenvalues of <R> and the helical period."""
    Rm = np.einsum("c,cjab->ab", w, M[:, :, :3, :3]) / M.shape[1]
    v = (T.X[T.Og] - T.X[T.O4p]) / 10.0; l2 = v @ v
    ev = np.linalg.eigvals(Rm)
    corr = lambda k: (v @ np.linalg.matrix_power(Rm, k) @ v) / l2
    I = np.eye(3)
    Cinf = 1 + 2 * (v @ Rm @ np.linalg.inv(I - Rm) @ v) / l2
    cp = ev[np.abs(ev.imag) > 1e-8]
    period = 2 * np.pi / abs(np.angle(cp[0])) if len(cp) else None
    out = dict(Cinf=float(Cinf), lam=[float(abs(e)) for e in ev], period_bonds=period,
               corr=[float(corr(k)) for k in range(12)], Cn={})
    for N in nlist:
        n = N - 1
        out["Cn"][N] = float(1 + 2.0 / n * sum((n - k) * corr(k) for k in range(1, n)))
    return out
