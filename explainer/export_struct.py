#!/usr/bin/env python3
"""2D structural formula of methyl alpha-maltoside with the two torsion bonds marked, in the style of IUPAC-IUB
1981 (Pure Appl. Chem. 55, 1269), Fig. 4: named atoms O5', C1', O4, C4, C5; the bond C1'-O4 (phi) and the bond
O4-C4 (psi) coloured, each with a curved arrow. chem env.  -> struct.svg (inlined into the page by build.py)"""
import re, math
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import sys
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "cheminf")); sys.argv = ["x"]
from chain_builder import SMILES
NAMES = {7: "C4", 4: "C5", 8: "O4", 9: "C1'", 10: "O5'"}            # atom order of maltose_a.xyz (scripts/figures_v2.py)
m = Chem.AddHs(Chem.MolFromSmiles(SMILES))
for a in m.GetAtoms():
    a.SetIntProp("orig", a.GetIdx())
    if a.GetIdx() in NAMES: a.SetProp("atomLabel", NAMES[a.GetIdx()])
m2 = Chem.RemoveHs(m); new = {a.GetIntProp("orig"): a.GetIdx() for a in m2.GetAtoms()}; AllChem.Compute2DCoords(m2)
W, H = 640, 360
d = rdMolDraw2D.MolDraw2DSVG(W, H); o = d.drawOptions(); o.bondLineWidth = 2; o.padding = 0.12; o.clearBackground = False
o.minFontSize = 17; o.maxFontSize = 22; o.highlightBondWidthMultiplier = 14
bphi = m2.GetBondBetweenAtoms(new[9], new[8]).GetIdx(); bpsi = m2.GetBondBetweenAtoms(new[8], new[7]).GetIdx()
BLUE, ORANGE = (0.16, 0.47, 0.84), (0.92, 0.41, 0.2)
rdMolDraw2D.PrepareAndDrawMolecule(d, m2, highlightAtoms=[], highlightBonds=[bphi, bpsi], highlightBondColors={bphi: BLUE, bpsi: ORANGE})
P = {k: d.GetDrawCoords(new[k]) for k in (4, 7, 8, 9, 10)}; allp = [d.GetDrawCoords(i) for i in range(m2.GetNumAtoms())]
d.FinishDrawing(); svg = d.GetDrawingText()
def arrow(a, b, label, col):
    mx, my = (a.x + b.x) / 2, (a.y + b.y) / 2; ux, uy = b.x - a.x, b.y - a.y; L = math.hypot(ux, uy); ux, uy = ux / L, uy / L; nx, ny = -uy, ux
    side = sum(1 if (p.x - mx) * nx + (p.y - my) * ny > 0 else -1 for p in allp)          # put the label on the emptier side
    s = -1 if side > 0 else 1
    lx, ly = mx + s * nx * 52, my + s * ny * 52
    return (f"<line x1='{mx + s*nx*9:.1f}' y1='{my + s*ny*9:.1f}' x2='{mx + s*nx*34:.1f}' y2='{my + s*ny*34:.1f}' style='stroke:{col};stroke-width:1.6'/>"
            f"<text x='{lx:.1f}' y='{ly + 10:.1f}' text-anchor='middle' style='font:italic 600 32px Georgia,serif;fill:{col}'>{label}</text>")
extra = arrow(P[9], P[8], "φ", "var(--blue)") + arrow(P[8], P[7], "ψ", "var(--orange)")
svg = svg.replace("</svg>", extra + "</svg>")
svg = re.sub(r"<\?xml[^>]*\?>\s*", "", svg); svg = re.sub(r"<rect[^>]*class='background'[^>]*>(</rect>)?", "", svg)
for hx, var in (("#2877D6", "var(--blue)"), ("#EA6833", "var(--orange)")):
    svg = svg.replace(f"fill:{hx};fill-rule:evenodd;fill-opacity:1;stroke:{hx}", f"fill:{var};fill-rule:evenodd;fill-opacity:.3;stroke:{var}").replace(hx, var)
svg = svg.replace("#000000", "var(--ink)").replace("#FF0000", "var(--atomO)")
svg = re.sub(r"<svg ", f"<svg role='img' aria-label='Methyl alpha-maltoside with the bonds of phi and psi marked' ", svg, count=1)
svg = re.sub(r"width='\d+px' height='\d+px'", "width='100%'", svg, count=1)
(HERE / "struct.svg").write_text(svg); print("bytes", len(svg)); print(sorted(set(re.findall(r"#[0-9A-Fa-f]{6}", svg))))
