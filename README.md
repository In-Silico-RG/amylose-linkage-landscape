# amylose-linkage-landscape

Code and data for the paper

> Y. Ruiz Rocha and A. F. Combariza, *Conformational Energy Landscape of the α(1→4) Linkage
> and the Dimensions of Amylose in Water*, submitted to *J. Phys. Chem. B* (2026).

**What this is.** How far an amylose chain spreads out in water depends on how each glucose
unit is joined to the next. We computed the energy of that joint (the α(1→4) glycosidic
linkage of methyl α-maltoside) over its two rotation angles, φ and ψ, with the quantum-chemical
method GFN2-xTB, and turned the resulting map into the dimensions of long chains, exactly
(transfer matrix) and by Monte Carlo sampling. This repository holds the scans, the validated
maps, the codes and the numbers behind every table and figure of the paper.

## Layout

| folder | contents |
|---|---|
| `cheminf/scan2/` | φ,ψ scan in implicit water, ALPB model (the paper's main map): one folder per φ row, `xtbscan.log` holds the 36 optimised frames (ψ = −120° + 10°·k) with energies |
| `cheminf/scan_gbsa/`, `cheminf/scan_vac/` | the same scan with GBSA water and in vacuum (sensitivity maps) |
| `cheminf/scan2_10deg.dat`, `scan_gbsa_10deg.dat`, `scan_vac_10deg.dat` | the validated maps, one line per cell (see below) |
| `cheminf/scan_10deg_*_INVALID.dat`, `scan_10deg.dat` | earlier scans rejected during validation (kept as record; see the SI) |
| `cheminf/*.py`, `*.sh` | chain builder, frame validation, map analysis, transfer matrix and Monte Carlo, DFT benchmark |
| `cheminf/template*/`, `*.xyz` | optimised maltose templates used to build rigid-residue chains |
| `cheminf/phipsi_*.npy` | (φ,ψ) pairs measured in the MD runs (SC6, SC18, DC6) |
| `scripts/` | analysis of the MD trajectories and the paper figures |
| `resultados/` | numerical results (JSON), one file per analysis |

**Map file format** (`*_10deg.dat`), whitespace-separated:
`φ_target ψ_target E_xtb(Hartree) φ_obtained ψ_obtained valid(1/0) reason`.
A cell is valid when the frame keeps the connectivity of methyl α-maltoside, both rings are
⁴C₁ chairs and both torsions are within 5° of their targets (`cheminf/validate_frames.py`).
Invalid cells are the ones where holding the angles forced a ring to deform; they lie at
energies above about 8 kcal/mol.

## Reproducing

Environment used: Python 3.12, NumPy, SciPy, Matplotlib, RDKit 2026.03, xtb 6.7.1,
PySCF 2.14, MDAnalysis 2.9, panedr 0.8.

- Map statistics: `cd cheminf && python map_analysis.py scan2_10deg.dat`
- Exact chain statistics and Monte Carlo chains: `cd cheminf && python mc_chains.py --help`
- DFT benchmark (hours per point): `cd cheminf && python dft_benchmark.py --small --xc r2scan`
- Figures: `python scripts/figures_v2.py`

Some shell scripts and the MD-analysis scripts contain absolute paths of the machine where they
were run (trajectory locations, the xtb binary); edit them before rerunning.

## Not included

The molecular dynamics inputs and production trajectories (CHARMM36/TIP3P, GROMACS 2023.3,
4.8 GB) are available from the corresponding author on request.

## Licence

Code (`*.py`, `*.sh`): MIT, see `LICENSE`. Data (scans, maps, results): CC BY 4.0, see
`LICENSE-DATA`.

## Contact

Aldo F. Combariza, Grupo de Investigación *in silico*, Universidad de Sucre, Colombia —
aldo.combariza@unisucre.edu.co
