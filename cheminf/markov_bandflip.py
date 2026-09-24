#!/usr/bin/env python3
"""Band-flip correlation between neighbouring linkages (referee r5 W4, 2026-09-24).
Two-state (syn/anti) first-order Markov version of the exact residue-frame RIS of ris_exact:
per-state mean rotations R_S, R_A (Boltzmann-weighted within the state), transition matrix
P(s'|s), stationary start. <R_1...R_k> = u Q^(k-1) [I;I], u = [pi_S R_S, pi_A R_A],
Q_(s,s') = P(s'|s) R_s'. Independent linkages (P(s'|s) = pi_s') reproduce ris_exact exactly.
Exclusion: P(A|A) = 0 with the same marginal p (P(A|S) = p/(1-p)).
Clustering: P(A|A) = 0.27 (SC40, SI) with the same marginal (P(A|S) = p(1-c)/(1-p)).
The band-flip population p is varied by a uniform shift of every anti cell, as in D-F9.
Output ../resultados/markov_bandflip.json"""
import json, sys
import numpy as np
sys.argv = ["x"]
import mc_chains as mc
from chain_builder import Template
kT = 0.0019872 * 303.15
T = Template(str(mc.HERE / "maltose_a_84_-145.xyz")); rng = np.random.default_rng(1)
pp, w0, E = mc.load_map("scan2_10deg.dat"); M = mc.precompute(T, pp, rng); M[:, :, :3, 3] /= 10
psiH = ((pp[:, 1] + 117.8 + 180) % 360) - 180; syn = np.abs(psiH) <= 90
Rc = M[:, :, :3, :3].mean(axis=1)                       # per-cell rotation, jitter variants averaged
v = (T.X[T.Og] - T.X[T.O4p]) / 10.0; l2 = v @ v; I3 = np.eye(3)

def markov_cinf(w, P):
    """w: cell weights; P: 2x2 transition matrix [[S->S, S->A], [A->S, A->A]]."""
    pS, pA = w[syn].sum(), w[~syn].sum()
    RS = np.einsum("c,cab->ab", w * syn, Rc) / pS; RA = np.einsum("c,cab->ab", w * ~syn, Rc) / pA
    ev, evec = np.linalg.eig(P.T); pi = np.real(evec[:, np.argmin(abs(ev - 1))]); pi /= pi.sum()
    R = [RS, RA]
    u = np.hstack([pi[0] * RS, pi[1] * RA])                # 3x6
    Q = np.block([[P[s, t] * R[t] for t in range(2)] for s in range(2)])   # 6x6
    J = np.vstack([I3, I3])                                # 6x3
    corr = lambda k: v @ (u @ np.linalg.matrix_power(Q, k - 1) @ J) @ v / l2
    Cinf = 1 + 2 * v @ (u @ np.linalg.inv(np.eye(6) - Q) @ J) @ v / l2
    return float(Cinf), float(pi[1]), [float(corr(k)) for k in (1, 2, 3)]

rows = []
for delta in [-1.5, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
    w = np.exp(-(E + np.where(syn, 0.0, delta)) / kT); w /= w.sum(); p = w[~syn].sum()
    ref = mc.ris_exact(M, w, T, nlist=(100,))["Cinf"]
    P_ind = np.array([[1 - p, p], [1 - p, p]])
    P_exc = np.array([[1 - p / (1 - p), p / (1 - p)], [1.0, 0.0]])
    c = 0.27; P_clu = np.array([[1 - p * (1 - c) / (1 - p), p * (1 - c) / (1 - p)], [1 - c, c]])
    ci, pi_i, _ = markov_cinf(w, P_ind); ce, pi_e, cc = markov_cinf(w, P_exc); cl, pi_c, _ = markov_cinf(w, P_clu)
    rows.append(dict(delta=delta, p_anti=float(p), Cinf_ris_exact=ref, Cinf_independent=ci,
                     Cinf_exclusion=ce, change_pct=100 * (ce / ci - 1), p_check_excl=pi_e, corr123_excl=cc,
                     Cinf_cluster027=cl, change_pct_cluster=100 * (cl / ci - 1), p_check_clu=pi_c))
    print(f"delta {delta:+5.2f}  p {100*p:5.1f}%  ris_exact {ref:6.2f}  indep {ci:6.2f}  exclusion {ce:6.2f} ({100*(ce/ci-1):+5.1f}%)  cluster {cl:6.2f} ({100*(cl/ci-1):+5.1f}%)  pi_A {100*pi_e:4.1f}/{100*pi_c:4.1f}%")
json.dump(rows, open("../resultados/markov_bandflip.json", "w"), indent=1)
