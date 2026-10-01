#!/usr/bin/env python3
"""Hydroxyl-arrangement structures for the explainer (chem env): for each of the seven valley cells the
all-atom structure as scanned (map frame, re-optimised: start tagged 'xtb') and the lowest of the ~300
sampled arrangements, both GFN2-xTB/ALPB, aligned on the heavy atoms of ring 1 of the template; the bond
list; the valley profiles of oh_compare.json.  -> oh.json (merged into the page by build.py)"""
import sys, json, re
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; CH = HERE.parent / "cheminf"
sys.path.insert(0, str(CH)); sys.argv = ["x"]
from chain_builder import Template, SMILES, kabsch, read_xyz
from rdkit import Chem
T = Template(str(CH / "maltose_a_84_-145.xyz")); m = Chem.AddHs(Chem.MolFromSmiles(SMILES)); ri = m.GetRingInfo()
ringA = [list(r) for r in ri.AtomRings() if T.C1p in r][0]; ringB = [list(r) for r in ri.AtomRings() if T.C4 in r][0]
bonds = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in m.GetBonds()]
summ = json.load(open(CH / "oh_sampling" / "xtb_summary.json")); H = 627.509
def load(cell, start):
    f = CH / "oh_sampling" / f"c_100_{cell}" / f"{start}.opt.xyz"; sym, X = read_xyz(f)
    E = float(re.search(r"energy:\s+(\S+)", f.read_text().split("\n")[1]).group(1))
    R, t = kabsch(X[ringA], T.X[ringA]); return (X @ R.T + t), E, sym
cells = {}
for psi in range(-170, -100, 10):
    s = summ[f"100,{psi}"]; tags = dict(l.split() for l in (CH / "oh_sampling" / f"c_100_{psi}" / "starts.txt").read_text().split("\n") if l)
    scan = [k for k, v in tags.items() if v == "xtb"][0]; best = s["kept"][0]["start"]
    Xs, Es, sym = load(psi, scan); Xb, Eb, _ = load(psi, best)
    cells[str(psi)] = dict(scan=[round(float(x), 3) for x in Xs.ravel()], best=[round(float(x), 3) for x in Xb.ravel()], dE=round((Es - Eb) * H, 2))
    print(psi, "as scanned is", cells[str(psi)]["dE"], "kcal/mol above the lowest arrangement found")
prof = json.load(open(HERE.parent / "resultados" / "oh_compare.json"))["profiles"]
out = dict(sym=sym, bonds=bonds, ringA=ringA, ringB=ringB, cells=cells,
           prof={k: [round(x, 2) for x in prof[k][:7]] for k in ("xtb map frame", "xtb best network", "TZVP-D3 // xtb map frame", "TZVP-D3 // 3c, best of 6")})
(HERE / "oh.json").write_text(json.dumps(out, separators=(",", ":"))); print("bytes", (HERE / "oh.json").stat().st_size)
