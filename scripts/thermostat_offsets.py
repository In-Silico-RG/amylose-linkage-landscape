#!/usr/bin/env python3
"""Solute and solvent temperatures of the production runs (separate v-rescale groups).
Reads the GROMACS .edr files with panedr (chem env); discards the first 2 ns, as in the
analysis; ΔT = <T-SOLU> - <T-SOLV>, standard error over four 2-ns blocks.
Output: ../resultados/thermostat_offsets.json
    /media/aldo/Aldo/envs/chem/bin/python thermostat_offsets.py
"""
import json
from pathlib import Path
import numpy as np
import panedr

ROOT = Path("/media/aldo/Aldo/sims-back-07-06-2025/yurimar")
RUNS = {"SC6": "SC/6SC-prod.edr", "SC12": "SC/12SC-prod.edr", "SC18": "SC/18SC-prod.edr",
        "SC24": "SC/24SC-prod.edr", "SC40": "SC/40SC-prod.edr", "DC6": "DC/6DC-prod.edr"}
out = {}
for name, f in RUNS.items():
    df = panedr.edr_to_df(str(ROOT / f))
    d = df[df.Time >= 2000]
    dT = (d["T-SOLU"] - d["T-SOLV"]).to_numpy()
    blocks = [b.mean() for b in np.array_split(dT, 4)]
    out[name] = dict(T_solute=float(d["T-SOLU"].mean()), T_solvent=float(d["T-SOLV"].mean()),
                     dT=float(dT.mean()), dT_se=float(np.std(blocks, ddof=1) / 2), n=len(d))
    print(f"{name:5s} T_solu {out[name]['T_solute']:.2f}  T_solv {out[name]['T_solvent']:.2f}  "
          f"dT {out[name]['dT']:.2f} ± {out[name]['dT_se']:.2f} K")
json.dump(out, open(Path(__file__).resolve().parent.parent / "resultados" / "thermostat_offsets.json", "w"), indent=1)
