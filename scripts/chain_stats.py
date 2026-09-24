#!/usr/bin/env python3
"""Virtual-bond chain statistics for amylose trajectories (Project C, JPCB paper).

Re-implementation of the procedures in Biofisica_Polimeros_Cadenas_ES/SI-es.tex
(the original Aug-2026 scripts were lost). One node per residue; per strand.

    python3 chain_stats.py SC6            # system key from SYSTEMS
    python3 chain_stats.py SC6 --node C1
Writes ../resultados/<system>_<node>.json
"""
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
import MDAnalysis as mda

warnings.filterwarnings("ignore")
DATA = Path("/media/aldo/Aldo/sims-back-07-06-2025/yurimar")
SYSTEMS = {
    "SC6":  (DATA / "SC/6SC-prod.tpr",  DATA / "SC/6SC-prod.xtc"),
    "SC12": (DATA / "SC/12SC-prod.tpr", DATA / "SC/12SC-prod.xtc"),
    "SC18": (DATA / "SC/18SC-prod.tpr", DATA / "SC/18SC-prod.xtc"),
    "SC24": (DATA / "SC/24SC-prod.tpr", DATA / "SC/24SC-prod.xtc"),
    "SC40": (DATA / "SC/40SC-prod.tpr", DATA / "SC/40SC-prod.xtc"),
    # the proposal's DC6 is the local run (.trr, 10 ps); the .xtc is the independent cluster run (2 ps)
    "DC6":  (DATA / "DC/6DC-prod.gro",  DATA / "DC/6DC-prod.trr"),
    "DC6xtc": (DATA / "DC/6DC-prod.gro", DATA / "DC/6DC-prod.xtc"),
}
T_START_PS = 2000.0   # discard first 2 ns
DT_PS = 20.0          # one configuration every 20 ps -> 401 frames
BLOCK_PS = 2000.0     # 2 ns blocks for standard errors
RING = "C1 C2 C3 C4 C5 O5"
LP_CUTOFF = 0.2      # tangent-fit window: initial points with C > 0.2


def strands(u):
    """Residues of the carbohydrate, split into strands (chains) in sequence order."""
    res = u.select_atoms("resname AGLC").residues
    if len(set(res.segids)) > 1:
        groups = [res[res.segids == s] for s in dict.fromkeys(res.segids)]
    else:
        # .gro has no segments: split where residue numbering restarts or chains break
        groups, cur = [], [res[0]]
        for a, b in zip(res[:-1], res[1:]):
            linked = np.linalg.norm(a.atoms.select_atoms("name O4").positions[0]
                                    - b.atoms.select_atoms("name C1").positions[0]) < 2.0
            if b.resid == a.resid + 1 and linked:
                cur.append(b)
            else:
                groups.append(cur); cur = [b]
        groups.append(cur)
    return [list(g) for g in groups]


def node_positions(residues, node, box):
    """Node coordinates in Angstrom; box in Angstrom."""
    if node == "ring":
        out = []
        for r in residues:
            p = r.atoms.select_atoms("name " + RING).positions
            d = p - p[0]
            d -= box * np.round(d / box)        # ring atoms can straddle the box edge
            out.append(p[0] + d.mean(0))
        return np.array(out)
    return np.array([r.atoms.select_atoms("name " + node).positions[0] for r in residues])


def make_chain_whole(x, box):
    """Unwrap a chain of nodes by minimum image between consecutive nodes (orthorhombic box)."""
    d = np.diff(x, axis=0)
    d -= box * np.round(d / box)
    return np.vstack([x[:1], x[:1] + np.cumsum(d, axis=0)])


def frame_stats(x):
    b = np.diff(x, axis=0)                      # nm
    blen = np.linalg.norm(b, axis=1)
    bh = b / blen[:, None]
    n = len(b)
    R = np.linalg.norm(x[-1] - x[0])
    Rg2 = np.mean(np.sum((x - x.mean(0)) ** 2, axis=1))
    corr = np.array([np.mean(np.sum(bh[: n - k] * bh[k:], axis=1)) for k in range(n)])
    D = np.linalg.norm(x[:, None] - x[None], axis=2) ** 2
    rm2 = np.array([np.mean(np.diagonal(D, m)) for m in range(1, n + 1)])
    # local frame on bond i (i >= 1): e1 along b_i, e2 from b_{i-1} orthogonalised
    F = []
    for i in range(1, n):
        e1 = bh[i]
        e2 = bh[i - 1] - np.dot(bh[i - 1], e1) * e1
        e2 /= np.linalg.norm(e2)
        F.append(np.column_stack([e1, e2, np.cross(e1, e2)]))
    T = [F[j].T @ F[j + 1] for j in range(len(F) - 1)]
    return dict(l=blen.mean(), l2=np.mean(blen ** 2), R=R, R2=R * R, Rg=np.sqrt(Rg2),
                Rg2=Rg2, corr=corr, rm2=rm2, Tsum=np.sum(T, axis=0), Tcount=len(T))


def block_se(values, times):
    nblocks = int((times[-1] - T_START_PS) // BLOCK_PS)
    # the final frame (t = 10 ns) joins the last full block
    blocks = np.minimum(((times - T_START_PS) // BLOCK_PS).astype(int), nblocks - 1)
    means = np.array([values[blocks == i].mean(0) for i in np.unique(blocks)])
    return means.std(0, ddof=1) / np.sqrt(len(means))


def run(system, node):
    top, trj = SYSTEMS[system]
    u = mda.Universe(str(top), str(trj))
    chains = strands(u)
    stride = max(1, int(round(DT_PS / u.trajectory.dt)))   # .trr of DC6 is saved every 50 ps
    rec, times = [], []
    for ts in u.trajectory[::stride]:
        if ts.time < T_START_PS - 1e-6:
            continue
        box = ts.dimensions[:3] / 10.0
        rec.append([frame_stats(make_chain_whole(node_positions(c, node, ts.dimensions[:3]) / 10.0, box))
                    for c in chains])
        times.append(ts.time)
    times = np.array(times)
    N = len(chains[0]); n = N - 1
    out = dict(system=system, node=node, n_strands=len(chains), N=N, n=n,
               n_frames=len(times), t0=float(times[0]), t1=float(times[-1]))
    # per-frame values averaged over strands
    def series(key):
        return np.array([np.mean([s[key] for s in fr], axis=0) for fr in rec])
    for key in ("l", "R", "R2", "Rg", "Rg2", "corr", "rm2"):
        v = series(key)
        out[key] = v.mean(0).tolist()
        out[key + "_se"] = block_se(v, times).tolist()
    l = out["l"]; L = n * l
    out["L"] = L
    out["Cn"] = out["R2"] / (n * l ** 2)
    out["Cm"] = (np.array(out["rm2"]) / (np.arange(1, n + 1) * l ** 2)).tolist()
    out["Rg_rod"] = l * np.sqrt((N ** 2 - 1) / 12.0)
    # policy (AFC 2026-09-19): rms Rg = sqrt<Rg2>, the same average as the rod value.
    # The Aug-2026 proposal used the mean <Rg> (SC18: 0.50 vs 0.52); kept as *_legacy.
    out["Rg_over_rod"] = np.sqrt(out["Rg2"]) / out["Rg_rod"]
    out["Rg_over_rod_continuous"] = np.sqrt(out["Rg2"]) / (L / np.sqrt(12.0))
    out["Rg_over_rod_legacy"] = out["Rg"] / out["Rg_rod"]
    # Kratky-Porod fit to tangent correlation over the whole chain, s = k l
    from scipy.optimize import curve_fit
    k = np.arange(n); c = np.array(out["corr"])
    (lp,), _ = curve_fit(lambda s, lp: np.exp(-s / lp), k * l, c, p0=[1.0],
                         bounds=(1e-3, 1e3))
    # decided 2026-09-19 (AFC): the nonlinear all-k fit is the reported tangent lp; the
    # Aug-2026 rule below matched SC6 only (SC18: 0.80 vs 0.53) and uses 2-3 points on short chains
    out["lp_tangent"] = float(lp)
    # legacy: log-linear fit over the initial decay, points with C > 0.2 (SC6 = 0.86 nm)
    first = np.argmax(c <= LP_CUTOFF) if np.any(c <= LP_CUTOFF) else n
    slope = np.polyfit(k[:first] * l, np.log(c[:first]), 1)[0]
    out["lp_tangent_c02"] = float(-1.0 / slope)
    out["lp_tangent_c02_npoints"] = int(first)

    # Benoit-Doty inversion of the virtual-chain Rg on the physical branch lp <= L
    from scipy.optimize import brentq
    def bd(lp):
        x = L / lp
        return L * lp / 3 - lp ** 2 + 2 * lp ** 3 / L * (1 - (1 - np.exp(-x)) / x)
    Rg2 = out["Rg2"]              # policy (AFC 2026-09-19): the formula is for <Rg2>
    grid = np.linspace(1e-3, L, 4000)
    vals = bd(grid) - Rg2
    idx = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
    out["lp_BD"] = float(brentq(lambda z: bd(z) - Rg2, grid[idx[0]], grid[idx[0] + 1])) if len(idx) else None
    out["BD_rod_limit_Rg2"] = float(bd(L))
    # legacy (Aug 2026): <Rg>^2 in place of <Rg2>; reproduces the proposal (SC18 0.69 nm)
    Rg2l = out["Rg"] ** 2
    vl = bd(grid) - Rg2l
    il = np.where(np.sign(vl[:-1]) != np.sign(vl[1:]))[0]
    out["lp_BD_legacy"] = float(brentq(lambda z: bd(z) - Rg2l, grid[il[0]], grid[il[0] + 1])) if len(il) else None

    # internal distances: <r^2(s)> = 2 lp s at large s
    m = np.arange(1, n + 1); rm2 = np.array(out["rm2"])
    # window m = 3..n (reproduces SC6 = 0.83 and SC18 = 0.63 nm; the earlier upper-half
    # rule reproduced SC6 only)
    a = 3
    out["lp_internal"] = float(np.polyfit(m[a - 1:] * l, rm2[a - 1:], 1)[0] / 2)
    out["lp_internal_window"] = [a, n]

    # scaling exponent from <r_m^2> ~ m^(2 nu)
    def nu(a, b):
        return float(np.polyfit(np.log(m[a - 1:b]), np.log(rm2[a - 1:b]), 1)[0] / 2)
    out["nu_short"] = nu(1, min(6, n))
    out["nu_long"] = nu(6, n) if n >= 10 else None

    # RIS transfer matrix
    Tsum = sum(fr_s["Tsum"] for fr in rec for fr_s in fr)
    Tcnt = sum(fr_s["Tcount"] for fr in rec for fr_s in fr)
    T = Tsum / Tcnt
    I = np.eye(3)
    corr_ris = [np.linalg.matrix_power(T, kk)[0, 0] for kk in range(n)]
    Cn_ris = 1 + 2.0 / n * sum((n - kk) * corr_ris[kk] for kk in range(1, n))
    ev = np.linalg.eigvals(T)
    lam = ev[np.argmax(np.abs(ev))]
    out["RIS"] = dict(steps=int(Tcnt), T=T.tolist(), corr=corr_ris, Cn=float(Cn_ris),
                      Cinf=float(((I + T) @ np.linalg.inv(I - T))[0, 0]),
                      eig=[[float(e.real), float(e.imag)] for e in ev],
                      lam_max=float(abs(lam)))
    cplx = ev[np.abs(ev.imag) > 1e-8]
    if len(cplx):
        theta = abs(np.angle(cplx[0]))
        out["RIS"]["period_bonds"] = float(2 * np.pi / theta)
        out["RIS"]["period_nm"] = float(2 * np.pi / theta * l)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("system", choices=SYSTEMS)
    ap.add_argument("--node", default="O4", choices=["O4", "C1", "ring"])
    a = ap.parse_args()
    res = run(a.system, a.node)
    dest = Path(__file__).resolve().parent.parent / "resultados" / f"{a.system}_{a.node}.json"
    dest.write_text(json.dumps(res, indent=1, default=float))
    r = res
    print(f"{r['system']} node={r['node']} strands={r['n_strands']} N={r['N']} frames={r['n_frames']} "
          f"t={r['t0']:.0f}-{r['t1']:.0f} ps")
    print(f"  l={r['l']:.3f}  L={r['L']:.2f}  <R>={r['R']:.2f}  sqrt<R2>={np.sqrt(r['R2']):.2f}  "
          f"<Rg>={r['Rg']:.2f}  sqrt<Rg2>={np.sqrt(r['Rg2']):.3f}  <R2>={r['R2']:.2f}  <Rg2>={r['Rg2']:.3f}")
    print(f"  Cn={r['Cn']:.2f}  Rg/rod={r['Rg_over_rod']:.2f} (continuous {r['Rg_over_rod_continuous']:.2f})"
          f"  shape R2/Rg2={r['R2']/r['Rg2']:.2f}  lp_tangent={r['lp_tangent']:.2f} nm (all-k nonlinear; legacy C>0.2 rule {r['lp_tangent_c02']:.2f}, {r['lp_tangent_c02_npoints']} pts)")
    print("  corr k=0..:", " ".join(f"{v:.2f}" for v in r["corr"]))
    print("  Cm  m=1..:", " ".join(f"{v:.2f}" for v in r["Cm"]))
    print(f"  lp_BD={r['lp_BD']}  lp_internal={r['lp_internal']:.2f} (m{r['lp_internal_window']})"
          f"  nu_short={r['nu_short']:.3f}  nu_long={r['nu_long']}")
    ris = r["RIS"]
    print(f"  RIS steps={ris['steps']} Cn={ris['Cn']:.2f} Cinf={ris['Cinf']:.2f} |lam|max={ris['lam_max']:.3f}"
          f" eig={[complex(round(a, 3), round(b, 3)) for a, b in ris['eig']]}"
          f" period={ris.get('period_nm')}")
    print("  RIS corr k=0..:", " ".join(f"{v:.2f}" for v in ris["corr"]))
