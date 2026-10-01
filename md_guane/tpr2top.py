#!/usr/bin/env python3
"""Rebuild a GROMACS topology from `gmx dump -s prod.tpr` (the CHARMM-GUI inputs of the 2025 amylose runs
are lost; only the production .tpr files survive). Needed for true replicas (new velocities), AFC 2026-09-25.

Writes every interaction with explicit parameters, so no force-field include is needed:
  [defaults] comb-rule 1 (c6/c12), full [nonbond_params] matrix from the tpr's LJ_SR table,
  bonds (1), Urey-Bradley angles (5), proper dihedrals (9, one line per Fourier term),
  H-bond constraints (1), explicit LJ-14 pairs (1), all tpr exclusions, SETTLE water.
Validated by `mdrun -rerun` of the start coordinates with the original and the rebuilt tpr (validate.sh).

    python tpr2top.py dump.txt out.top
"""
import re, sys
from collections import defaultdict

FT = re.compile(r"functype\[(\d+)\]=(\w+), (.*)")
KV = re.compile(r"(\w+)=\s*([-+0-9.eE]+)")


def parse(path):
    L = open(path).read().split("\n")
    ff, atnr, fudge, mols, blocks, cur, sec = {}, None, 1.0, [], [], None, None
    for i, l in enumerate(L):
        s = l.strip()
        if m := FT.match(s):
            ff[int(m.group(1))] = (m.group(2), {k: float(v) for k, v in KV.findall(m.group(3))}); continue
        if s.startswith("atnr="): atnr = int(s.split("=")[1]); continue
        if s.startswith("fudgeQQ"): fudge = float(s.split("=")[1]); continue
        if m := re.match(r'moltype\s+=\s+(\d+) "(\S+)"', s): blocks.append([m.group(2), None]); continue
        if s.startswith("#molecules") and blocks and blocks[-1][1] is None: blocks[-1][1] = int(s.split("=")[1]); continue
        if re.match(r"moltype \(\d+\):", s):
            cur = dict(name=None, atoms=[], names=[], types=[], res=[], excls=[], il=defaultdict(list)); mols.append(cur); sec = None; continue
        if cur is None: continue
        if s.startswith("name=") and cur["name"] is None: cur["name"] = s.split('"')[1]; continue
        if m := re.match(r"atom\[\s*\d+\]=\{type=\s*(\d+).*m=\s*(\S+), q=\s*(\S+),.*resind=\s*(\d+)", s):
            cur["atoms"].append((int(m.group(1)), float(m.group(2).rstrip(",")), float(m.group(3).rstrip(",")), int(m.group(4)))); continue
        if m := re.match(r'atom\[\s*\d+\]=\{name="(\S+)"\}', s): cur["names"].append(m.group(1)); continue
        if m := re.match(r'type\[\s*\d+\]=\{name="(\S+)",', s): cur["types"].append(m.group(1)); continue
        if m := re.match(r'residue\[\d+\]=\{name="(\S+)"', s): cur["res"].append(m.group(1)); continue
        if m := re.match(r"excls\[(\d+)\]\[num=\d+\]=\{(.*)", s):
            txt, j = m.group(2), i
            while "}" not in txt: j += 1; txt += L[j].strip()
            cur["excls"].append([int(v) for v in txt.split("}")[0].split(",") if v.strip()]); continue
        if m := re.match(r"(Bond|U-B|Proper Dih\.|LJ-14|Constraint|Settle|Improper Dih\.|Angle|Per\. Imp\. Dih\.|CMAP Dih\.):$", s): sec = m.group(1); continue
        if s == "iatoms:" or s.startswith("nr:"): continue
        if re.match(r"[A-Za-z0-9 .\-]+:$", s): sec = None; continue
        if sec and (m := re.match(r"\d+ type=(\d+) \((\w+)\)\s+(.*)", s)):
            cur["il"][sec].append((int(m.group(1)), [int(v) for v in m.group(3).split()]))
    return ff, atnr, fudge, mols, blocks


def write(ff, atnr, fudge, mols, blocks, out):
    tname = {}
    for mt in mols:
        for (t, *_), n in zip(mt["atoms"], mt["types"]): tname[t] = n
    znum = {}
    lj = {(i, j): ff[i * atnr + j][1] for i in range(atnr) for j in range(atnr)}
    assert all(ff[k][0] == "LJ_SR" for k in range(atnr * atnr))
    for mt in mols:
        for bad in ("Improper Dih.", "Angle", "Per. Imp. Dih.", "CMAP Dih."):
            assert not mt["il"].get(bad), f"{mt['name']}: {bad} not handled"
    o = ["; rebuilt from a .tpr dump by tpr2top.py (explicit parameters; see header of that script)",
         "[ defaults ]", "; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ", f"1 1 no 1.0 {fudge}", "",
         "[ atomtypes ]", "; name mass charge ptype c6 c12"]
    masses = {}
    for mt in mols:
        for (t, m, q, r) in mt["atoms"]: masses[t] = m
    for t in range(atnr):
        o.append(f"{tname[t]:8s} {masses[t]:10.5f} 0.0 A {lj[t, t]['c6']:.8e} {lj[t, t]['c12']:.8e}")
    o += ["", "[ nonbond_params ]"]
    for i in range(atnr):
        for j in range(i, atnr):
            o.append(f"{tname[i]:8s} {tname[j]:8s} 1 {lj[i, j]['c6']:.8e} {lj[i, j]['c12']:.8e}")
    for mt in mols:
        o += ["", "[ moleculetype ]", f"{mt['name']} 3", "", "[ atoms ]"]
        for k, ((t, m, q, r), n) in enumerate(zip(mt["atoms"], mt["names"])):
            o.append(f"{k+1:6d} {tname[t]:8s} {r+1:4d} {mt['res'][r]:6s} {n:6s} {k+1:6d} {q:10.6f} {m:10.5f}")
        il = mt["il"]
        if il["Bond"]:
            o += ["", "[ bonds ]"] + [f"{a+1} {b+1} 1 {ff[t][1]['b0A']:.6e} {ff[t][1]['cbA']:.6e}" for t, (a, b) in il["Bond"]]
        if il["U-B"]:
            o += ["", "[ angles ]"] + [f"{a+1} {b+1} {c+1} 5 {p['thetaA']:.8e} {p['kthetaA']:.8e} {p['r13A']:.8e} {p['kUBA']:.8e}"
                                       for t, (a, b, c) in il["U-B"] for p in [ff[t][1]]]
        if il["Proper Dih."]:
            o += ["", "[ dihedrals ]"] + [f"{a+1} {b+1} {c+1} {d+1} 9 {p['phiA']:.6f} {p['cpA']:.8e} {int(p['mult'])}"
                                          for t, (a, b, c, d) in il["Proper Dih."] for p in [ff[t][1]]]
        if il["LJ-14"]:
            o += ["", "[ pairs ]"] + [f"{a+1} {b+1} 1 {ff[t][1]['c6A']:.8e} {ff[t][1]['c12A']:.8e}" for t, (a, b) in il["LJ-14"]]
        if il["Constraint"]:
            o += ["", "[ constraints ]"] + [f"{a+1} {b+1} 1 {ff[t][1]['dA']:.8e}" for t, (a, b) in il["Constraint"]]
        if il["Settle"]:
            (t, (a, b, c)), = il["Settle"]; p = ff[t][1]
            o += ["", "[ settles ]", f"{a+1} 1 {p['doh']:.8e} {p['dhh']:.8e}"]
        o += ["", "[ exclusions ]"] + [" ".join(str(v + 1) for v in [k] + [e for e in ex if e != k])
                                        for k, ex in enumerate(mt["excls"]) if len(ex) > 1]
    o += ["", "[ system ]", "rebuilt from tpr", "", "[ molecules ]"] + [f"{n} {c}" for n, c in blocks]
    open(out, "w").write("\n".join(o) + "\n")


if __name__ == "__main__":
    write(*parse(sys.argv[1]), sys.argv[2])
