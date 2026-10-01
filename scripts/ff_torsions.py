#!/usr/bin/env python3
"""Explicit torsion terms of CHARMM36 and GLYCAM06j about the two glycosidic bonds (AFC 2026-10-01).
Read from the topologies used in the long runs (md_guane/6SC_rebuilt.top, md_guane/glycam_SC6/sc6.top): every
proper dihedral whose central bond is C1'-O4 (phi axis) or O4-C4 (psi axis), terms k[1 + cos(n theta - delta)].
Also the sum of those terms as a function of phi or psi, with the other substituents placed at the offsets they
have on the template (measured from cheminf/maltose_a_84_-145.xyz through the chain builder).
    -> ../resultados/ff_torsions.json, ../manuscript_JPCB/tab_ff_torsions.tex   (chem env, for the offsets)"""
import re, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; MD = HERE.parent / "md_guane"; RES = HERE.parent / "resultados"
KJ = 4.184


def parse(top):
    sec = None; atoms = {}; dih = []
    for l in open(top):
        l = l.split(";")[0].strip()
        if not l: continue
        m = re.match(r"\[\s*(\w+)\s*\]", l)
        if m: sec = m.group(1); continue
        p = l.split()
        if sec == "atoms" and len(p) >= 5 and p[0].isdigit(): atoms[int(p[0])] = (p[1], int(p[2]), p[4])
        elif sec == "dihedrals" and len(p) >= 8 and p[0].isdigit(): dih.append(([int(x) for x in p[:4]], float(p[5]), float(p[6]) / KJ, int(p[7])))
    return atoms, dih


def terms(top, ra, rb):
    """ra: residue that gives C1 (primed ring of the paper); rb: residue that gives O4, C4."""
    atoms, dih = parse(top); idx = {(a[1], a[2]): i for i, a in atoms.items()}; out = {"phi": {}, "psi": {}}
    for ax, b, own in (("phi", (idx[(ra, "C1")], idx[(rb, "O4")]), ra), ("psi", (idx[(rb, "O4")], idx[(rb, "C4")]), rb)):
        for q, delta, k, n in dih:
            if {q[1], q[2]} == set(b):
                end = [i for i in (q[0], q[3]) if atoms[i][1] == own and atoms[i][2] not in ("C1", "C4", "O4")]
                end = end[0] if end else [i for i in (q[0], q[3]) if atoms[i][2] not in ("C1", "C4", "O4")][0]
                out[ax].setdefault(atoms[end][2], []).append(dict(n=n, k=round(k, 3), delta=delta))
    return out


def offsets():
    """dihedral of each substituent minus the paper's phi (O5') or psi (C5), on the rigid template"""
    sys.path.insert(0, str(HERE.parent / "cheminf")); sys.argv = ["x"]
    from chain_builder import Template, SMILES, build_chain, dihedral
    from rdkit import Chem
    T = Template(str(HERE.parent / "cheminf" / "maltose_a_84_-145.xyz")); m = Chem.AddHs(Chem.MolFromSmiles(SMILES)); X = T.X
    nb = lambda i: [n.GetIdx() for n in m.GetAtomWithIdx(i).GetNeighbors()]
    d = lambda a, b, c, e: np.degrees(dihedral(X[a], X[b], X[c], X[e])); w = lambda a: (a + 180) % 360 - 180
    C2p = [i for i in nb(T.C1p) if T.sym[i] == "C"][0]; H1p = [i for i in nb(T.C1p) if T.sym[i] == "H"][0]
    C3 = [i for i in nb(T.C4) if T.sym[i] == "C" and i != T.C5][0]; H4 = [i for i in nb(T.C4) if T.sym[i] == "H"][0]
    phi = d(T.O5p, T.C1p, T.Og, T.C4); psi = d(T.C1p, T.Og, T.C4, T.C5)
    return dict(phi=dict(O5=0.0, C2=w(d(C2p, T.C1p, T.Og, T.C4) - phi), H1=w(d(H1p, T.C1p, T.Og, T.C4) - phi)),
                psi=dict(C5=0.0, C3=w(d(T.C1p, T.Og, T.C4, C3) - psi), H4=w(d(T.C1p, T.Og, T.C4, H4) - psi)))


def total(tt, off, x):
    """sum of the explicit terms about one bond when the reference dihedral is x (degrees)"""
    return sum(t["k"] * (1 + np.cos(np.radians(t["n"] * (x + off[a]) - t["delta"]))) for a, L in tt.items() for t in L)


if __name__ == "__main__":
    ff = {"CHARMM36": terms(MD / "6SC_rebuilt.top", 3, 2), "GLYCAM06": terms(MD / "glycam_SC6" / "sc6.top", 4, 3)}
    off = offsets(); x = np.arange(-180, 180, 1.0); out = dict(terms=ff, offsets=off, profiles={})
    for f in ff:
        out["profiles"][f] = {}
        for ax in ("phi", "psi"):
            e = total(ff[f][ax], off[ax], x); at = lambda v: float(total(ff[f][ax], off[ax], np.array([v]))[0] - e.min())
            out["profiles"][f][ax] = dict(range=float(e.max() - e.min()), at_min=float(x[e.argmin()]), at_max=float(x[e.argmax()]),
                                           values={str(v): round(at(v), 2) for v in ((60, 80, 90, 100, 120) if ax == "phi" else (-170, -150, -130, -110, 70))})
            print(f, ax, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out["profiles"][f][ax].items()})
    (RES / "ff_torsions.json").write_text(json.dumps(out, indent=1))
    lab = {"phi": {"O5": "O5$'$--C1$'$--O4--C4 ($\\varphi$)", "C2": "C2$'$--C1$'$--O4--C4", "H1": "H1$'$--C1$'$--O4--C4"},
           "psi": {"C5": "C1$'$--O4--C4--C5 ($\\psi$)", "C3": "C1$'$--O4--C4--C3", "H4": "C1$'$--O4--C4--H4"}}
    fmt = lambda L: ", ".join(f"{t['k']:.2f} ({t['n']}; {t['delta']:.0f}$^\\circ$)" for t in sorted(L, key=lambda t: t["n"])) if L else "---"
    rows = []
    for ax, head in (("phi", "about C1$'$--O4"), ("psi", "about O4--C4")):
        rows.append(f"\\emph{{{head}}} & & \\\\")
        for a in lab[ax]: rows.append(f"{lab[ax][a]} & {fmt(ff['CHARMM36'][ax].get(a, []))} & {fmt(ff['GLYCAM06'][ax].get(a, []))} \\\\")
        p = out["profiles"]; rows.append(f"sum of the three, range over a full turn & {p['CHARMM36'][ax]['range']:.1f} & {p['GLYCAM06'][ax]['range']:.1f} \\\\")
        if ax == "phi": rows.append("\\midrule")
    (HERE.parent / "manuscript_JPCB" / "tab_ff_torsions.tex").write_text("\n".join(rows) + "\n\\bottomrule\n"); print("\n".join(rows))
