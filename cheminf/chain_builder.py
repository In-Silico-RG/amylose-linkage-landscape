#!/usr/bin/env python3
"""Rigid-residue amylose chain builder for RIS / Monte Carlo (WP6, WP6b).

Residue geometry comes from an xtb-optimised maltose (template XYZ, atom order of maltose.xyz).
Each linkage is set by (phi, psi) = (O5'-C1'-O4-C4, C1'-O4-C4-C5); bond lengths and valence
angles at the bridge are taken from the template and kept fixed. Nodes of the virtual chain
are the O4 atoms (same convention as chain_stats.py).

    python3 chain_builder.py --test        # rigid helix at the B-amylose torsions (84, -145)
Library use: from chain_builder import Template, build_chain, helix_params
"""
import sys, argparse
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SMILES = "CO[C@@H]1O[C@H](CO)[C@@H](O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@H](O)[C@H]1O"   # methyl alpha-maltoside, atom order of maltose_a.xyz


def read_xyz(f):
    L = open(f).read().split("\n")
    n = int(L[0]); sym, xyz = [], []
    for l in L[2:2 + n]:
        p = l.split(); sym.append(p[0]); xyz.append([float(v) for v in p[1:4]])
    return sym, np.array(xyz)


def nerf(a, b, c, bond, angle, torsion):
    """Place d given a-b-c, |cd| = bond, angle bcd, torsion abcd (radians)."""
    bc = c - b; bc /= np.linalg.norm(bc)
    n = np.cross(b - a, bc); n /= np.linalg.norm(n)
    m = np.cross(n, bc)
    d2 = np.array([-bond * np.cos(angle), bond * np.sin(angle) * np.cos(torsion),
                   bond * np.sin(angle) * np.sin(torsion)])
    return c + d2[0] * bc + d2[1] * m + d2[2] * n


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n; w = b2 - np.dot(b2, b1n) * b1n
    return np.arctan2(np.dot(np.cross(b1n, v), w), np.dot(v, w))


def angle(a, b, c):
    u, v = a - b, c - b
    return np.arccos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v))


def kabsch(P, Q):
    """Rotation R, translation t with R @ P_i + t ~ Q_i (P, Q: k x 3)."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc


class Template:
    """Atom bookkeeping from the SMILES graph (same atom order as the XYZ)."""
    def __init__(self, xyz_file):
        from rdkit import Chem
        m = Chem.AddHs(Chem.MolFromSmiles(SMILES))
        self.sym, self.X = read_xyz(xyz_file)
        assert m.GetNumAtoms() == len(self.sym)
        ri = m.GetRingInfo()
        ringO = lambda a: [n for n in a.GetNeighbors() if n.GetSymbol() == "O" and ri.NumAtomRings(n.GetIdx())]
        for o in m.GetAtoms():
            if o.GetSymbol() != "O" or ri.NumAtomRings(o.GetIdx()): continue
            nb = [n for n in o.GetNeighbors() if n.GetSymbol() == "C" and ri.NumAtomRings(n.GetIdx())]
            if len(nb) != 2: continue
            c1 = [c for c in nb if ringO(c)]; c4 = [c for c in nb if not ringO(c)]
            if len(c1) == 1 and len(c4) == 1:
                Og, C1p, C4 = o.GetIdx(), c1[0].GetIdx(), c4[0].GetIdx()
                O5p = ringO(c1[0])[0].GetIdx()
                C5 = [n for n in c4[0].GetNeighbors() if n.GetSymbol() == "C" and ringO(n)][0].GetIdx()
                C5p = [n for n in m.GetAtomWithIdx(O5p).GetNeighbors() if n.GetIdx() != C1p][0].GetIdx()
        self.Og, self.C1p, self.C4, self.O5p, self.C5, self.C5p = Og, C1p, C4, O5p, C5, C5p
        # Repeating unit = residue A (non-reducing ring: it carries the alpha C1'-O linkage
        # natively) plus the glycosidic O. Residue B (reducing end, OMe) is only used for the
        # bridge angle C1-O-C4. The chain node is A's O4' (its hydroxyl O), which coincides
        # with the previous residue's glycosidic O once the chain is built.
        seen, stack = {C1p}, [C1p]
        while stack:
            a = stack.pop()
            for n in m.GetAtomWithIdx(a).GetNeighbors():
                j = n.GetIdx()
                if j != Og and j not in seen:
                    seen.add(j); stack.append(j)
        self.res = sorted(seen | {Og})
        C4p = [n.GetIdx() for n in m.GetAtomWithIdx(C5p).GetNeighbors()
               if n.GetSymbol() == "C" and ri.NumAtomRings(n.GetIdx())][0]
        O4p = [n.GetIdx() for n in m.GetAtomWithIdx(C4p).GetNeighbors()
               if n.GetSymbol() == "O" and not ri.NumAtomRings(n.GetIdx())][0]
        self.C4p, self.O4p = C4p, O4p
        X = self.X
        self.a_C1OC4 = angle(X[C1p], X[Og], X[C4])          # bridge valence angle (template)
        self.b_OC4 = np.linalg.norm(X[C4p] - X[O4p])        # A's own C4'-O4' bond
        self.a_OC4C5 = angle(X[O4p], X[C4p], X[C5p])
        self.b_C4C5 = np.linalg.norm(X[C5p] - X[C4p])
        self.triad = [O4p, C4p, C5p]
        self.heavy = [i for i in self.res if self.sym[i] != "H"]


def build_chain(T, phipsi_deg):
    """phipsi_deg: (N-1) x 2 array of linkage torsions. Returns list of residue coordinate
    arrays (template atom order) and the N x 3 O4 node array."""
    X = T.X; res = [X[T.res].copy()]
    idx = {a: k for k, a in enumerate(T.res)}       # template index -> column in residue array
    O4s = [X[T.O4p]]
    for phi, psi in np.deg2rad(np.asarray(phipsi_deg, float)):
        cur = res[-1]
        O5, C1, Og = cur[idx[T.O5p]], cur[idx[T.C1p]], cur[idx[T.Og]]
        C4n = nerf(O5, C1, Og, T.b_OC4, T.a_C1OC4, phi)      # torsion O5'-C1'-O4-C4 = phi
        C5n = nerf(C1, Og, C4n, T.b_C4C5, T.a_OC4C5, psi)    # torsion C1'-O4-C4-C5 = psi
        P = X[T.triad]; Q = np.array([Og, C4n, C5n])
        R, t = kabsch(P, Q)
        res.append((R @ X[T.res].T).T + t)
        O4s.append(Og)                                       # next residue's O4 = this Og
    return res, np.array(O4s)


def helix_params(res, T):
    """Screw parameters residue i -> i+1: residues per turn, rise, pitch, handedness."""
    idx = {a: k for k, a in enumerate(T.res)}
    k = [idx[a] for a in T.heavy]
    R, t = kabsch(res[0][k], res[1][k])
    theta = np.arccos((np.trace(R) - 1) / 2)
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]); ax /= np.linalg.norm(ax)
    # rise = component of the displacement of any point along the axis (same for all points)
    p = res[0][k].mean(0); q = res[1][k].mean(0)
    rise = np.dot(q - p, ax)
    n = 2 * np.pi / theta
    hand = "left" if rise * theta < 0 else "right"      # sign convention: axis from R, rise along it
    # make rise positive and read handedness from the rotation sense about +axis
    return dict(residues_per_turn=n, rise_nm=abs(rise) / 10, pitch_nm=abs(rise) * n / 10,
                twist_deg=np.degrees(theta), handedness=hand)


def chain_stats_from_nodes(x):
    """l, tangent correlation, C_n, and the RIS transfer matrix <T> (same as chain_stats.py)."""
    b = np.diff(x, axis=0); bl = np.linalg.norm(b, axis=1); bh = b / bl[:, None]; n = len(b)
    corr = np.array([np.mean(np.sum(bh[:n - k] * bh[k:], axis=1)) for k in range(n)])
    R2 = np.sum((x[-1] - x[0]) ** 2)
    F = []
    for i in range(1, n):
        e1 = bh[i]; e2 = bh[i - 1] - np.dot(bh[i - 1], e1) * e1; e2 /= np.linalg.norm(e2)
        F.append(np.column_stack([e1, e2, np.cross(e1, e2)]))
    Tm = np.mean([F[j].T @ F[j + 1] for j in range(len(F) - 1)], axis=0)
    return dict(l=bl.mean(), corr=corr, Cn=R2 / (n * bl.mean() ** 2), T=Tm)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--test", action="store_true")
    ap.add_argument("--template", default=str(HERE / "maltose_a_84_-145.xyz"))
    a = ap.parse_args()
    T = Template(a.template)
    print(f"template: O4'-C4' {T.b_OC4:.3f} A, angle C1-O-C4 {np.degrees(T.a_C1OC4):.1f} deg, virtual bond O4'-Og {np.linalg.norm(T.X[T.O4p]-T.X[T.Og])/10:.3f} nm")
    print(f"template torsions: phi {np.degrees(dihedral(*T.X[[T.O5p, T.C1p, T.Og, T.C4]])):.1f}  psi {np.degrees(dihedral(*T.X[[T.C1p, T.Og, T.C4, T.C5]])):.1f}")
    for phi, psi, label in ((84, -145, "B-amylose duplex strand (Imberty & Perez 1988)"),
                            (100, -130, "syn region, typical"), (-60, -40, "test point")):
        res, O4 = build_chain(T, [(phi, psi)] * 23)
        h = helix_params(res, T); s = chain_stats_from_nodes(O4 / 10)
        ev = np.linalg.eigvals(s["T"]); cp = ev[np.abs(ev.imag) > 1e-6]
        per = 2 * np.pi / abs(np.angle(cp[0])) if len(cp) else float("nan")
        print(f"({phi:4d},{psi:5d}) {label}: n={h['residues_per_turn']:.2f} res/turn, rise {h['rise_nm']:.3f} nm, "
              f"pitch {h['pitch_nm']:.2f} nm, {h['handedness']}-handed | l(O4-O4)={s['l']:.3f} nm  C_24={s['Cn']:.2f} "
              f"| <T> eig |.|={np.round(np.abs(ev),3)} period={per:.2f} bonds")
        print("   corr k=0..7:", np.round(s["corr"][:8], 2))
