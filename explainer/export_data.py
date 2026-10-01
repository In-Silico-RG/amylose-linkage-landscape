#!/usr/bin/env python3
"""Data for the visual explainer two_numbers.html (2026-10-01, AFC: "I still can not picture this").
Exports from the paper's own objects (ALPB map, rigid template, cell transforms with the paper's jitter
seed) everything the page needs to recompute the exact transfer-matrix C_inf in the browser and to draw
chains: per valid cell phi, psi, E (kcal/mol), mean rotation over jitter variants (exact C_inf) and the
centre transform (drawing); a fine row of transforms along the valley at phi = 100 (ideal helix); the
band-flip transform (90, 70); template coordinates of one residue; the virtual bond.
    /media/aldo/Aldo/envs/chem/bin/python export_data.py   ->  data.json (embedded into the page by build.py)"""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; CH = HERE.parent / "cheminf"
sys.path.insert(0, str(CH)); sys.argv = ["x"]
import os; os.chdir(CH)
import mc_chains as mc
from chain_builder import Template, SMILES
from rdkit import Chem

T = Template(str(CH / "maltose_a_84_-145.xyz")); rng = np.random.default_rng(1)
pp, w, E = mc.load_map("scan2_10deg.dat"); M = mc.precompute(T, pp, rng)          # translations in Angstrom
r = mc.ris_exact(M / np.array([1, 1, 1, 1])[None, None, :, None] * 1.0, w, T, nlist=(100,))
print("check C_inf", r["Cinf"])
tr = lambda A: [round(float(x), 5) for x in A[:3, :3].ravel()] + [round(float(x), 3) for x in A[:3, 3]]
cells = dict(phi=[int(p) for p in pp[:, 0]], psi=[int(p) for p in pp[:, 1]], E=[round(float(e), 3) for e in E],
             Rbar=[[round(float(x), 5) for x in M[c, :, :3, :3].mean(0).ravel()] for c in range(len(pp))],
             T0=[tr(M[c, 0]) for c in range(len(pp))])
psis = np.arange(-190, -92.5, 2.5)
valley = dict(psi=[float(p) for p in psis], T=[tr(mc.step_transform(T, np.radians(100.0), np.radians(p))) for p in psis])
m = Chem.AddHs(Chem.MolFromSmiles(SMILES)); ri = m.GetRingInfo()
ring = [list(rg) for rg in ri.AtomRings() if T.C1p in rg][0]
C6 = [n.GetIdx() for n in m.GetAtomWithIdx(T.C5p).GetNeighbors() if n.GetSymbol() == "C" and not ri.NumAtomRings(n.GetIdx())][0]
O6 = [n.GetIdx() for n in m.GetAtomWithIdx(C6).GetNeighbors() if n.GetSymbol() == "O"][0]
xyz = lambda i: [round(float(x), 3) for x in T.X[i]]
out = dict(cells=cells, valley=valley, flip=tr(mc.step_transform(T, np.radians(90.0), np.radians(70.0))),
           ring=[xyz(i) for i in ring], ringO=ring.index(T.O5p), O4=xyz(T.O4p), Og=xyz(T.Og), C6=xyz(C6), O6=xyz(O6),
           v=[float(x) for x in (T.X[T.Og] - T.X[T.O4p]) / 10.0], kT=0.0019872 * 303.15, Cinf_check=r["Cinf"])
(HERE / "data.json").write_text(json.dumps(out, separators=(",", ":")))
print("cells", len(pp), "bytes", (HERE / "data.json").stat().st_size, "ring atoms", [T.sym[i] for i in ring])
