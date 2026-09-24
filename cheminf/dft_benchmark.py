#!/usr/bin/env python3
"""WP-M2d: DFT single points on selected cells of the ALPB map (referee request A1, 2026-09-23).
Geometries: the xtb-optimised frames of scan2 (row_<phi>/xtbscan.log, frame k <-> psi = -120 + 10k).
Level: PySCF, density-fitted RKS (--xc r2scan|pbe|b3lyp), def2-TZVP or def2-SVP (--small), ddCOSMO water (eps 78.4) unless --gas, 8 threads.
Output: ../resultados/dft_benchmark.json (appended cell by cell; safe to rerun).
    python dft_benchmark.py [--small] [--cells "100,-150 100,-120 ..."]
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
DEFAULT = ["100,-170", "100,-160", "100,-150", "100,-140", "100,-130", "100,-120", "100,-110",
           "80,-150", "90,-150", "110,-150", "120,-150",
           "90,70", "60,50", "100,130", "100,140", "100,120", "100,-20", "100,-30", "100,-10",
           "160,-110", "60,-170", "170,-60"]


def frame(phi, psi):
    log = HERE / "scan2" / f"row_{phi}" / "xtbscan.log"
    L = open(log).read().split("\n"); n = int(L[0]); k = ((psi + 120) // 10) % 36
    blk = L[k * (n + 2): k * (n + 2) + n + 2]
    atoms = [(l.split()[0], [float(v) for v in l.split()[1:4]]) for l in blk[2:2 + n]]
    e = float(blk[1].split()[1]) if "energy" in blk[1] else None
    return atoms, e


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--small", action="store_true"); ap.add_argument("--cells", default=None); ap.add_argument("--xc", default="r2scan"); ap.add_argument("--grid", type=int, default=2); ap.add_argument("--gas", action="store_true")
    a = ap.parse_args()
    from pyscf import gto, dft, lib
    lib.num_threads(8)
    basis = "def2-svp" if a.small else "def2-tzvp"
    out = HERE.parent / "resultados" / (f"dft_benchmark_{a.xc}_{basis}{'_gas' if a.gas else ''}.json")
    res = json.load(open(out)) if out.exists() else {}
    cells = (a.cells.split() if a.cells else DEFAULT)
    valid = {(int(r[0]), int(r[1])) for r in (l.split() for l in open(HERE / "scan2_10deg.dat")) if len(r) >= 6 and r[5] == "1"}
    for c in cells:
        phi, psi = (int(v) for v in c.split(","))
        if c in res: continue
        if (phi, psi) not in valid: print(c, "not a valid cell, skipped"); continue
        atoms, exs = frame(phi, psi)
        mol = gto.M(atom=[(s, tuple(x)) for s, x in atoms], basis=basis, charge=0, spin=0, verbose=0)
        t0 = time.time()
        mf = dft.RKS(mol).density_fit(); mf.xc = a.xc; mf.grids.level = a.grid; mf.conv_tol = 1e-7
        if not a.gas: mf = mf.ddCOSMO(); mf.with_solvent.eps = 78.4
        e = mf.kernel()
        res[c] = dict(phi=phi, psi=psi, E_dft=float(e), E_xtb=exs, converged=bool(mf.converged), basis=basis, xc=a.xc, solvent=("none" if a.gas else "ddCOSMO"), seconds=time.time() - t0)
        json.dump(res, open(out, "w"), indent=1)
        print(f"{c}: E_dft {e:.6f} Eh (xtb {exs})  {time.time() - t0:.0f} s  conv {mf.converged}", flush=True)


if __name__ == "__main__":
    main()
