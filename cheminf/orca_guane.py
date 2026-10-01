#!/usr/bin/env python3
"""DFT benchmark r3 on GUANE (ORCA 6.1.1), reviewers r6/r7 (Kimi W2, DeepSeek W1): larger basis,
dispersion, DFT-relaxed geometries. Run on the GUANE cluster (SC3-UIS), 2026-09-25.

Per cell (the 20 cells of dft_benchmark_r2scan_def2-svp.json), four ORCA steps in one Slurm job:
  1 sp_tzvp   r2SCAN-D3(BJ)/def2-TZVP, CPCM(water), on the xtb/ALPB frame
  2 sp_svp    r2SCAN/def2-SVP, CPCM(water), no D3, on the xtb frame (bridge to the PySCF set)
  3 opt_3c    r2SCAN-3c, CPCM(water), phi and psi constrained at the xtb frame values
  4 sp_opt    r2SCAN-D3(BJ)/def2-TZVP, CPCM(water), on the step-3 geometry
Torsions, 0-based: phi = O5'-C1'-O4-C4 = 10 9 8 7, psi = C1'-O4-C4-C5 = 9 8 7 4 (checked on three frames).

    python orca_guane.py build              -> orca_guane/<cell>/{frame.xyz,*.inp}, orca_guane/job.sh
    python orca_guane.py collect            -> ../resultados/dft_orca_r2scan.json
"""
import sys, json, re
from pathlib import Path
from dft_benchmark import frame, HERE

OUT = HERE / "orca_guane"
NPROCS, MAXCORE = 12, 3500
SOLV = ""   # CPCM(water) keyword sets the solvent
PAL = f"%pal nprocs {NPROCS} end\n%maxcore {MAXCORE}\n"
TORS = "%geom\n  Constraints\n    { D 10 9 8 7 C }\n    { D 9 8 7 4 C }\n  end\nend\n"

STEPS = {
    "1_sp_tzvp": ("! r2SCAN D3BJ def2-TZVP def2/J RIJCOSX TightSCF CPCM(water)\n", "frame.xyz", ""),
    "2_sp_svp":  ("! r2SCAN def2-SVP def2/J RIJCOSX TightSCF CPCM(water)\n", "frame.xyz", ""),
    "3_opt_3c":  ("! r2SCAN-3c TightSCF Opt CPCM(water)\n", "frame.xyz", TORS),
    "4_sp_opt":  ("! r2SCAN D3BJ def2-TZVP def2/J RIJCOSX TightSCF CPCM(water)\n", "3_opt_3c.xyz", ""),
}

JOB = """#!/bin/bash
#SBATCH -J amy_{tag}
#SBATCH -p legacy
#SBATCH -N 1
#SBATCH --ntasks={n}
#SBATCH --mem=45G
#SBATCH -o slurm-%j.out
# usage: sbatch job.sh <cell directory>; runs the four ORCA steps in order, skipping finished ones
module load orca/6.1.1
cd "$1"
ORCA=$(which orca)
for s in 1_sp_tzvp 2_sp_svp 3_opt_3c 4_sp_opt; do
  grep -q "ORCA TERMINATED NORMALLY" $s.out 2>/dev/null && continue
  $ORCA $s.inp > $s.out 2>&1
  grep -q "ORCA TERMINATED NORMALLY" $s.out || {{ echo "$s failed"; exit 1; }}
done
echo done > DONE
"""


def build():
    cells = list(json.load(open(HERE.parent / "resultados" / "dft_benchmark_r2scan_def2-svp.json")))
    for c in cells:
        phi, psi = (int(v) for v in c.split(","))
        d = OUT / f"c_{phi}_{psi}"; d.mkdir(parents=True, exist_ok=True)
        atoms, _ = frame(phi, psi)
        (d / "frame.xyz").write_text(f"{len(atoms)}\ncell {c} xtb/ALPB scan2 frame\n" +
                                     "".join(f"{s} {x[0]:.6f} {x[1]:.6f} {x[2]:.6f}\n" for s, x in atoms))
        for name, (kw, xyz, extra) in STEPS.items():
            (d / f"{name}.inp").write_text(kw + PAL + SOLV + extra + f"* xyzfile 0 1 {xyz}\n")
    (OUT / "job.sh").write_text(JOB.format(tag="dft", n=NPROCS))
    (OUT / "cells.txt").write_text("\n".join(f"c_{c.replace(',', '_')}" for c in cells) + "\n")
    print(len(cells), "cells ->", OUT)


def collect():
    res = {}
    for d in sorted(OUT.glob("c_*")):
        _, phi, psi = d.name.split("_"); r = dict(phi=int(phi), psi=int(psi))
        for name in STEPS:
            f = d / f"{name}.out"
            if f.exists() and "ORCA TERMINATED NORMALLY" in f.read_text(errors="ignore"):
                e = re.findall(r"FINAL SINGLE POINT ENERGY\s+(\S+)", f.read_text(errors="ignore"))
                r[name] = float(e[-1])
        res[f"{phi},{psi}"] = r
    out = HERE.parent / "resultados" / "dft_orca_r2scan.json"
    json.dump(res, open(out, "w"), indent=1); print(out, sum(len(v) - 2 for v in res.values()), "energies")


if __name__ == "__main__":
    {"build": build, "collect": collect}[sys.argv[1]]()
