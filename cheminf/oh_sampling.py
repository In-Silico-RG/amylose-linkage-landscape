#!/usr/bin/env python3
"""Hydroxyl-network sampling on the seven syn-valley cells (phi = 100, psi = -170 ... -110).
AFC 2026-09-25 ("YES, STRONG"): the DFT-relaxed valley of DFT r3 depended on which OH network each cell
started from (two compact cells relaxed 3.5 kcal/mol more than the rest, with a rearranged network).

  gen      -> oh_sampling/<cell>/s_XXX.xyz starts: the cell's xtb frame, the r2SCAN-3c network of each of
              the seven cells transplanted onto it (OH and CH2OH dihedrals copied), and NRAND random
              networks (every hydroxyl H staggered at 60/180/300 deg + noise, every CH2OH at gg/gt/tg).
  collect  -> after the GUANE xtb step (job_oh_xtb.sh): validates every optimised frame (connectivity,
              4C1 rings, phi/psi within 5 deg; validate_frames.check), dedupes (0.05 kcal/mol and same
              H-bond set), writes the NKEEP lowest per cell to oh_sampling/dft/<cell>_kK/ with ORCA inputs
              (r2SCAN-3c constrained opt, then r2SCAN-D3(BJ)/def2-TZVP, CPCM water; as orca_guane.py).
  final    -> after the DFT step: lowest energy per cell at each level -> ../resultados/oh_sampling.json
  A second argument "anti" runs the same three steps on the two band-flip cells (90,70) and (60,50) in
  oh_sampling_anti/ (networks transplanted from the seven valley cells and the two anti cells).
"""
import sys, json, re
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
OUT = HERE / "oh_sampling"
CELLS = [(100, p) for p in range(-170, -100, 10)]
NRAND, NKEEP, SEED = 300, 6, 20260925
ANTI = [(90, 70), (60, 50)]                                     # band-flip minima of the benchmark (2026-09-30)
DONORS, RESULT = CELLS, "oh_sampling.json"
if len(sys.argv) > 2 and sys.argv[2] == "anti":                 # same protocol on the two band-flip cells: <cmd> anti
    DONORS, CELLS, OUT, SEED, RESULT = CELLS + ANTI, ANTI, HERE / "oh_sampling_anti", SEED + 1, "oh_sampling_anti.json"
PHI, PSI = (10, 9, 8, 7), (9, 8, 7, 4)                          # 0-based (checked on three frames)


def rxyz(f):
    L = Path(f).read_text().split("\n"); n = int(L[0])
    a = [l.split() for l in L[2:2 + n]]
    return [x[0] for x in a], np.array([[float(v) for v in x[1:4]] for x in a])


def wxyz(f, S, X, title=""):
    Path(f).write_text(f"{len(S)}\n{title}\n" + "".join(f"{s} {x[0]:.6f} {x[1]:.6f} {x[2]:.6f}\n" for s, x in zip(S, X)))


def dih(X, i, j, k, l):
    b0, b1, b2 = X[i] - X[j], X[k] - X[j], X[l] - X[k]; b1 = b1 / np.linalg.norm(b1)
    v = b0 - b0 @ b1 * b1; w = b2 - b2 @ b1 * b1
    return np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w))


def bonds(S, X):
    r = {"H": 0.32, "C": 0.76, "O": 0.66}
    D = np.linalg.norm(X[:, None] - X[None], axis=2)
    return {i: [j for j in range(len(S)) if j != i and D[i, j] < 1.2 * (r[S[i]] + r[S[j]])] for i in range(len(S))}


def rotors(S, X):
    """(axis_a, axis_b, moving atoms, reference atom for the dihedral) for every OH and CH2OH."""
    B = bonds(S, X); rot = []
    for h in range(len(S)):
        if S[h] == "H" and len(B[h]) == 1 and S[B[h][0]] == "O":
            o = B[h][0]; c = [a for a in B[o] if a != h][0]
            ref = [a for a in B[c] if a != o and S[a] != "H"][0]
            rot.append((c, o, [h], (h, o, c, ref)))                      # hydroxyl H about C-O
    for c6 in range(len(S)):                                             # CH2OH: C with 2 H, one OH-O, one C
        if S[c6] != "C": continue
        nb = B[c6]; hs = [a for a in nb if S[a] == "H"]; os_ = [a for a in nb if S[a] == "O"]; cs = [a for a in nb if S[a] == "C"]
        if len(hs) == 2 and len(os_) == 1 and len(cs) == 1 and any(S[a] == "H" for a in B[os_[0]]):
            o6, c5 = os_[0], cs[0]; ho = [a for a in B[o6] if S[a] == "H"]
            ring_o = [a for a in B[c5] if S[a] == "O"][0]
            rot.append((c5, c6, [o6] + ho, (o6, c6, c5, ring_o)))       # omega about C5-C6
    return rot


def set_dih(X, r, target):
    a, b, move, d = r; X = X.copy()
    ang = np.radians(target - dih(X, *d))
    k = X[b] - X[a]; k /= np.linalg.norm(k); P = X[move] - X[b]
    X[move] = X[b] + P * np.cos(ang) + np.cross(k, P) * np.sin(ang) + np.outer(P @ k, k) * (1 - np.cos(ang))
    return X


def gen():
    rng = np.random.default_rng(SEED)
    frames = {c: rxyz(HERE / f"orca_guane/c_{c[0]}_{c[1]}/frame.xyz") for c in CELLS}
    opts = {c: rxyz(HERE / f"orca_guane/c_{c[0]}_{c[1]}/3_opt_3c.xyz") for c in DONORS}
    S0, X0 = frames[CELLS[0]]; R = rotors(S0, X0)
    print(len(R), "rotors:", [("CH2OH" if len(r[2]) > 1 else "OH") for r in R])
    for c in CELLS:
        S, X = frames[c]; d = OUT / f"c_{c[0]}_{c[1]}"; d.mkdir(parents=True, exist_ok=True)
        starts = [("xtb", X)]
        for c2 in DONORS:                                                 # transplanted DFT networks
            Y = X.copy(); Z = opts[c2][1]
            for r in R: Y = set_dih(Y, r, dih(Z, *r[3]))
            starts.append((f"dft{c2[0]}_{c2[1]}" if DONORS is not CELLS else f"dft{c2[1]}", Y))
        for k in range(NRAND):
            Y = X.copy()
            for r in R:
                base = rng.choice([60.0, 180.0, 300.0]) if len(r[2]) == 1 else rng.choice([60.0, 180.0, 300.0])
                Y = set_dih(Y, r, base + rng.uniform(-20, 20))
            starts.append((f"rnd{k}", Y))
        with open(d / "starts.txt", "w") as fh:
            for i, (tag, Y) in enumerate(starts):
                wxyz(d / f"s_{i:03d}.xyz", S, Y, f"cell {c} start {tag}"); fh.write(f"s_{i:03d} {tag}\n")
        (d / "c.inp").write_text("$constrain\n  force constant=1.0\n"
                                 f"  dihedral: 11,10,9,8,{c[0]:.1f}\n  dihedral: 10,9,8,5,{c[1]:.1f}\n$end\n")
    print("starts per cell:", len(starts))


def hbset(S, X):
    """Set of intramolecular O-H...O contacts (donor O, acceptor O) with H...O < 2.5 A."""
    B = bonds(S, X); hb = set()
    for h in range(len(S)):
        if S[h] == "H" and len(B[h]) == 1 and S[B[h][0]] == "O":
            for o in range(len(S)):
                if S[o] == "O" and o != B[h][0] and np.linalg.norm(X[o] - X[h]) < 2.5: hb.add((B[h][0], o))
    return frozenset(hb)


def collect():
    sys.path.insert(0, str(HERE))
    from validate_frames import check
    H = 627.509; summary = {}
    DFT = OUT / "dft"; DFT.mkdir(exist_ok=True)
    for c in CELLS:
        d = OUT / f"c_{c[0]}_{c[1]}"; tags = dict(l.split() for l in (d / "starts.txt").read_text().split("\n") if l)
        rows = []
        for f in sorted(d.glob("s_*.opt.xyz")):
            S, X = rxyz(f); E = float(re.search(r"energy:\s+(\S+)", f.read_text().split("\n")[1]).group(1))
            ok, why = check(S, X); dp, ds = dih(X, *PHI) - c[0], dih(X, *PSI) - c[1]
            dp, ds = (dp + 180) % 360 - 180, (ds + 180) % 360 - 180
            if ok and abs(dp) < 5 and abs(ds) < 5: rows.append((E, f.name.split(".")[0], tags[f.name.split(".")[0]], hbset(S, X), S, X))
        rows.sort(key=lambda r: r[0]); keep = []
        for r in rows:
            if all(abs(r[0] - k[0]) * H > 0.05 or r[3] != k[3] for k in keep): keep.append(r)
            if len(keep) == NKEEP: break
        e0 = rows[0][0]
        summary[f"{c[0]},{c[1]}"] = dict(n_opt=len(list(d.glob("s_*.opt.xyz"))), n_valid=len(rows),
            xtb_frame=next((r[0] - e0) * H for r in rows if r[2] == "xtb") if any(r[2] == "xtb" for r in rows) else None,
            kept=[dict(start=r[1], tag=r[2], dE_xtb=(r[0] - e0) * H, nHB=len(r[3])) for r in keep])
        for k, r in enumerate(keep):
            kd = DFT / f"c_{c[0]}_{c[1]}_k{k}"; kd.mkdir(exist_ok=True)
            wxyz(kd / "frame.xyz", r[4], r[5], f"cell {c} {r[1]} {r[2]} xtb E {r[0]:.9f}")
            pal = "%pal nprocs 12 end\n%maxcore 3500\n"
            tors = "%geom\n  Constraints\n    { D 10 9 8 7 C }\n    { D 9 8 7 4 C }\n  end\nend\n"
            (kd / "3_opt_3c.inp").write_text("! r2SCAN-3c TightSCF Opt CPCM(water)\n" + pal + tors + "* xyzfile 0 1 frame.xyz\n")
            (kd / "4_sp_opt.inp").write_text("! r2SCAN D3BJ def2-TZVP def2/J RIJCOSX TightSCF CPCM(water)\n" + pal + "* xyzfile 0 1 3_opt_3c.xyz\n")
        s = summary[f"{c[0]},{c[1]}"]
        print(f"{c}: {s['n_opt']} optimised, {s['n_valid']} valid; xtb-frame network at {s['xtb_frame']:.2f} kcal/mol above the best;",
              "kept", [(k['tag'], round(k['dE_xtb'], 2), k['nHB']) for k in s['kept']])
    (OUT / "xtb_summary.json").write_text(json.dumps(summary, indent=1))


def final():
    H = 627.509; res = {}
    for c in CELLS:
        r = {}
        for kd in sorted((OUT / "dft").glob(f"c_{c[0]}_{c[1]}_k*")):
            for s in ("3_opt_3c", "4_sp_opt"):
                f = kd / f"{s}.out"
                if f.exists() and "ORCA TERMINATED NORMALLY" in f.read_text(errors="ignore"):
                    r.setdefault(s, {})[kd.name] = float(re.findall(r"FINAL SINGLE POINT ENERGY\s+(\S+)", f.read_text(errors="ignore"))[-1])
        res[f"{c[0]},{c[1]}"] = {s: dict(min=min(v.values()), argmin=min(v, key=v.get), all=v) for s, v in r.items()}
    for s in ("3_opt_3c", "4_sp_opt"):
        e0 = min(res[k][s]["min"] for k in res if s in res[k])
        print(s, " ".join(f"{k.split(',')[1]}:{(res[k][s]['min'] - e0) * H:.2f}" for k in res if s in res[k]))
    (HERE.parent / "resultados" / RESULT).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"gen": gen, "collect": collect, "final": final}[sys.argv[1]]()
