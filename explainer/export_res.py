#!/usr/bin/env python3
"""All atoms of the repeating unit for the explainer (chem env): residue A of the template (the unit the paper's
chain builder repeats) with its hydrogens, hydroxyl groups and the bridging oxygen on C1; element, name,
coordinates in the template frame and bonds.  -> res.json (merged by build.py as D.res)"""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; CH = HERE.parent / "cheminf"
sys.path.insert(0, str(CH)); sys.argv = ["x"]
from chain_builder import Template, SMILES
from rdkit import Chem
T = Template(str(CH / "maltose_a_84_-145.xyz")); m = Chem.AddHs(Chem.MolFromSmiles(SMILES)); ri = m.GetRingInfo()
nb = lambda i: [n.GetIdx() for n in m.GetAtomWithIdx(i).GetNeighbors()]
inring = lambda i: ri.NumAtomRings(i) > 0
C1, O5, C5, C4, O4, Og = T.C1p, T.O5p, T.C5p, T.C4p, T.O4p, T.Og
C2 = [i for i in nb(C1) if T.sym[i] == "C"][0]; C3 = [i for i in nb(C4) if T.sym[i] == "C" and i != C5][0]
C6 = [i for i in nb(C5) if T.sym[i] == "C" and not inring(i)][0]
Oof = lambda c: [i for i in nb(c) if T.sym[i] == "O" and not inring(i) and i != Og][0]
name = {C1: "C1", C2: "C2", C3: "C3", C4: "C4", C5: "C5", O5: "O5", C6: "C6", O4: "O4", Og: "O1", Oof(C2): "O2", Oof(C3): "O3", Oof(C6): "O6"}
for i in T.res:
    if T.sym[i] == "H":
        p = nb(i)[0]; name[i] = ("H" + name[p][1:]) if T.sym[p] == "C" else ("HO" + name[p][1:])
res = list(T.res); pos = {a: k for k, a in enumerate(res)}
bonds = [[pos[b.GetBeginAtomIdx()], pos[b.GetEndAtomIdx()]] for b in m.GetBonds() if b.GetBeginAtomIdx() in pos and b.GetEndAtomIdx() in pos]
idx = {name[a]: k for k, a in enumerate(res)}
out = dict(sym=[T.sym[a] for a in res], names=[name[a] for a in res], xyz=[[round(float(x), 3) for x in T.X[a]] for a in res], bonds=bonds, idx=idx)
(HERE / "res.json").write_text(json.dumps(out, separators=(",", ":")))
print(len(res), "atoms:", " ".join(out["names"])); print("bonds", len(bonds))
# consistency with data.json
D = json.loads((HERE / "data.json").read_text())
assert np.allclose(out["xyz"][idx["O4"]], D["O4"], atol=2e-3) and np.allclose(out["xyz"][idx["O1"]], D["Og"], atol=2e-3) and np.allclose(out["xyz"][idx["C1"]], D["ring"][D["ix"]["C1"]], atol=2e-3)
print("consistent with data.json")
