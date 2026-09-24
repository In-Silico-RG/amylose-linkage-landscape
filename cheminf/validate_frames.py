#!/usr/bin/env python3
"""Parse an xtb xtbscan.log (XYZ frames, energy in the comment line) and validate each frame:
connectivity identical to methyl alpha-maltoside, both rings 4C1 chairs, torsions within
5 deg of the requested (phi, psi). Prints: phi_req psi_req E phi psi valid(1/0) reason.
    python3 validate_frames.py xtbscan.log PHI_FIXED  (row scan: psi from -120 step 10)
    python3 validate_frames.py xtbscan.log --column   (column scan: phi from 100 step 10, psi -120)
Also: --extract K out.xyz  writes frame K.
"""
import sys, re
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from chain_builder import dihedral

SMI = "CO[C@@H]1O[C@H](CO)[C@@H](O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@H](O)[C@H]1O"
ref = Chem.AddHs(Chem.MolFromSmiles(SMI)); ri = ref.GetRingInfo()
REFB = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in ref.GetBonds()}
def rings():
    out = []
    for r in ri.AtomRings():
        O = [a for a in r if ref.GetAtomWithIdx(a).GetSymbol() == "O"][0]
        C1 = [n.GetIdx() for n in ref.GetAtomWithIdx(O).GetNeighbors()
              if any(x.GetSymbol() == "O" and not ri.NumAtomRings(x.GetIdx()) for x in n.GetNeighbors())][0]
        order = [O, C1]; cur = C1
        while len(order) < 6:
            cur = [n.GetIdx() for n in ref.GetAtomWithIdx(cur).GetNeighbors() if n.GetIdx() in r and n.GetIdx() not in order][0]
            order.append(cur)
        out.append(order)
    return out
RINGS = rings()
TOR = [int(v) - 1 for v in open(__file__.replace("validate_frames.py", "torsions_a.txt")).read().split()]

def frames(f):
    L = open(f).read().split("\n"); i = 0; out = []
    while i < len(L) and L[i].strip():
        n = int(L[i]); cm = L[i + 1]
        m = re.findall(r"-?\d+\.\d+", cm); E = float(m[0]) if m else float("nan")
        sym = [l.split()[0] for l in L[i + 2:i + 2 + n]]
        X = np.array([[float(v) for v in l.split()[1:4]] for l in L[i + 2:i + 2 + n]])
        out.append((E, sym, X)); i += n + 2
    return out

def check(sym, X):
    xyz = f"{len(sym)}\n\n" + "\n".join(f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in zip(sym, X))
    m = Chem.MolFromXYZBlock(xyz); rdDetermineBonds.DetermineConnectivity(m)
    B = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in m.GetBonds()}
    if B != REFB: return False, f"bonds {len(REFB - B)}-/{len(B - REFB)}+"
    for r in RINGS:
        t = [np.degrees(dihedral(*X[[r[i], r[(i + 1) % 6], r[(i + 2) % 6], r[(i + 3) % 6]]])) for i in range(6)]
        chair = all(40 < abs(v) < 75 for v in t) and all(np.sign(t[i]) != np.sign(t[(i + 1) % 6]) for i in range(6))
        if not (chair and t[0] > 0): return False, "ring"
    return True, "ok"

if __name__ == "__main__":
    f = sys.argv[1]
    if sys.argv[2] == "--extract":
        E, sym, X = frames(f)[int(sys.argv[3])]
        open(sys.argv[4], "w").write(f"{len(sym)}\n{E}\n" + "\n".join(f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in zip(sym, X)) + "\n")
        sys.exit()
    col = sys.argv[2] in ("--column", "--column-back"); back = sys.argv[2] == "--column-back"
    for k, (E, sym, X) in enumerate(frames(f)):
        phi = np.degrees(dihedral(*X[TOR[:4]])); psi = np.degrees(dihedral(*X[TOR[4:]]))
        if col: preq, sreq = (100 + (-10 if back else 10) * k + 180) % 360 - 180, -120
        else: preq, sreq = float(sys.argv[2]), (-120 + 10 * k + 180) % 360 - 180
        ok, why = check(sym, X)
        dp = (phi - preq + 180) % 360 - 180; ds = (psi - sreq + 180) % 360 - 180
        if abs(dp) > 5 or abs(ds) > 5: ok, why = False, f"drift {dp:.0f}/{ds:.0f}"
        print(f"{preq:.0f} {sreq:.0f} {E:.9f} {phi:.1f} {psi:.1f} {int(ok)} {why}")
