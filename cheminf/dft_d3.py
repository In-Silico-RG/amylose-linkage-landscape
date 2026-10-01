#!/usr/bin/env python3
"""D3(BJ) dispersion for the DFT benchmark cells (Kimi r6 / DeepSeek r5: 'no dispersion').
D3 depends only on the geometry, so it is added a posteriori to any r2SCAN single point on the same
xtb frame. Uses s-dftd3 1.4.0 (chem env), r2SCAN-D3(BJ) parameters built in, one thread per call.
Output: ../resultados/dft_d3_r2scan_bj.json  {cell: {phi, psi, E_d3}}  (Eh)
    python dft_d3.py [--cells "100,-150 ..."]   (default: every cell of the SVP benchmark file)
"""
import json, os, re, subprocess, tempfile, argparse
from pathlib import Path
from dft_benchmark import frame, HERE

S_DFTD3 = Path("/media/aldo/Aldo/envs/chem/bin/s-dftd3")


def e_d3(atoms):
    with tempfile.TemporaryDirectory() as d:
        xyz = Path(d) / "m.xyz"
        xyz.write_text(f"{len(atoms)}\n\n" + "".join(f"{s} {x[0]} {x[1]} {x[2]}\n" for s, x in atoms))
        out = subprocess.run([str(S_DFTD3), "--bj", "r2scan", "m.xyz"], cwd=d, capture_output=True, text=True,
                             env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True).stdout
    return float(re.search(r"Dispersion energy:\s+(\S+)", out).group(1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cells", default=None); a = ap.parse_args()
    svp = json.load(open(HERE.parent / "resultados" / "dft_benchmark_r2scan_def2-svp.json"))
    cells = a.cells.split() if a.cells else list(svp)
    out = HERE.parent / "resultados" / "dft_d3_r2scan_bj.json"
    res = json.load(open(out)) if out.exists() else {}
    for c in cells:
        phi, psi = (int(v) for v in c.split(","))
        res[c] = dict(phi=phi, psi=psi, E_d3=e_d3(frame(phi, psi)[0]))
        print(f"{c}: E_D3 {res[c]['E_d3']:.6f} Eh", flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
