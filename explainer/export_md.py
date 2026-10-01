#!/usr/bin/env python3
"""Snapshots of the 12-ring chain from one CHARMM36 replica for the explainer (MDAnalysis: ~/miniconda3/bin/python):
30 frames between 2 and 100 ns of SC12 r2; per frame, for each ring in the page's order (the ring whose C1
carries the bridging oxygen first), the ring atoms O5 C5 C4 C3 C2 C1, its O4, C6 and O6, with the whole chain
moved so that the first ring sits on the template ring; and phi, psi of each joint.  -> md.json"""
import json, warnings
from pathlib import Path
import numpy as np, MDAnalysis as mda
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; M = HERE.parent / "md_guane" / "rep_12SC"
D = json.loads((HERE / "data.json").read_text()); tpl = np.array(D["ring"])
full = mda.Universe(str(M / "start_2ns.gro")); u = mda.Merge(full.select_atoms("resname AGLC")); u.load_new(str(M / "r2" / "md.xtc"))
res = list(u.select_atoms("resname AGLC").residues)[::-1]            # page order: C1 of ring k is bonded to O4 of ring k+1
names = ["O5", "C5", "C4", "C3", "C2", "C1", "O4", "C6", "O6"]
ix = np.array([[r.atoms.select_atoms("name " + n).indices[0] for n in names] for r in res])
def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0); U, S, Vt = np.linalg.svd((P - pc).T @ (Q - qc)); d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T; return R, qc - R @ pc
def dih(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2; b1 = b1 / np.linalg.norm(b1); v = b0 - b0 @ b1 * b1; w = b2 - b2 @ b1 * b1
    return float(np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w)))
frames = []; nf = len(u.trajectory); pick = np.linspace(200, nf - 1, 30).astype(int)
for f in pick:
    ts = u.trajectory[f]; X = ts.positions.astype(float); box = ts.dimensions[:3].astype(float)
    o4 = X[ix[:, 6]]; d = np.diff(o4, axis=0); d -= box * np.round(d / box); O4 = np.vstack([o4[:1], o4[:1] + np.cumsum(d, 0)])
    P = X[ix] - o4[:, None, :]; P -= box * np.round(P / box); P += O4[:, None, :]                  # every ring whole around its O4
    R, t = kabsch(P[0, :6], tpl); P = P @ R.T + t
    tors = [[dih(P[k, 0], P[k, 5], P[k + 1, 6], P[k + 1, 2]), dih(P[k, 5], P[k + 1, 6], P[k + 1, 2], P[k + 1, 1])] for k in range(len(res) - 1)]
    bridge = [float(np.degrees(np.arccos(np.dot(*( (v / np.linalg.norm(v)) for v in (P[k, 5] - P[k + 1, 6], P[k + 1, 2] - P[k + 1, 6])))))) for k in range(len(res) - 1)]
    frames.append(dict(t=round(float(ts.time) / 1000, 1), X=[round(float(x), 2) for x in P.ravel()], tors=[[round(a, 1), round(b, 1)] for a, b in tors], bridge=[round(b, 1) for b in bridge]))
print("frames", len(frames), "mean bridge angle", np.mean([f["bridge"] for f in frames]), "mean phi psi", np.mean([f["tors"] for f in frames], axis=(0, 1)))
(HERE / "md.json").write_text(json.dumps(dict(names=names, frames=frames), separators=(",", ":"))); print("bytes", (HERE / "md.json").stat().st_size)
