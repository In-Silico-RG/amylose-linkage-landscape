#!/usr/bin/env python3
"""Relaxation diagnostics and intrachain contacts (WP1 additions, plan r3).

For each system (O4 virtual chain, per strand averaged):
  * time series of Rg and R^2 every DT_PS from t = 0 (full run) and their integrated
    autocorrelation time tau (sum of the normalised ACF up to its first zero crossing);
    N_eff = T / (2 tau) on the full run and on the 2-10 ns analysis window;
  * drift test: mean Rg over the first vs second half of the analysis window (block error);
  * contacts: heavy-atom pairs closer than 0.5 nm between residues more than two apart in
    sequence, per residue (SI-es.tex step 7). For duplexes, intra-strand and inter-strand
    counts are given separately.

    python3 relaxation.py SC18      -> ../resultados/SC18_relax.json
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chain_stats import SYSTEMS, T_START_PS, DT_PS, strands, node_positions, make_chain_whole

CUTOFF_A = 5.0     # contact cutoff, Angstrom
MIN_SEP = 3        # residues more than two apart


def acf_tau(x, dt):
    x = np.asarray(x) - np.mean(x)
    n = len(x)
    c = np.correlate(x, x, "full")[n - 1:] / (np.arange(n, 0, -1) * np.var(x))
    zero = np.argmax(c <= 0) if np.any(c <= 0) else n
    tau = (0.5 + c[1:zero].sum()) * dt        # integrated, in ps
    return tau, zero * dt


def run(system):
    top, trj = SYSTEMS[system]
    u = mda.Universe(str(top), str(trj))
    chains = strands(u)
    heavy = u.select_atoms("resname AGLC and not name H*")
    resid_of = heavy.resids
    # strand index of each heavy atom
    strand_of = np.empty(len(heavy), int)
    for si, ch in enumerate(chains):
        ids = {r.resindex for r in ch}
        strand_of[[i for i, a in enumerate(heavy) if a.resindex in ids]] = si
    stride = max(1, int(round(DT_PS / u.trajectory.dt)))
    t, Rg, R2, c_intra, c_inter = [], [], [], [], []
    for ts in u.trajectory[::stride]:
        box = ts.dimensions[:3]
        rg, r2 = [], []
        for ch in chains:
            x = make_chain_whole(node_positions(ch, "O4", box) / 10.0, box / 10.0)
            rg.append(np.sqrt(np.mean(np.sum((x - x.mean(0)) ** 2, 1))))
            r2.append(np.sum((x[-1] - x[0]) ** 2))
        Rg.append(np.mean(rg)); R2.append(np.mean(r2)); t.append(ts.time)
        pairs = capped_distance(heavy.positions, heavy.positions, CUTOFF_A, box=ts.dimensions,
                                return_distances=False)
        i, j = pairs[:, 0], pairs[:, 1]
        m = i < j
        i, j = i[m], j[m]
        same = strand_of[i] == strand_of[j]
        far = np.abs(resid_of[i] - resid_of[j]) >= MIN_SEP
        c_intra.append(np.sum(same & far)); c_inter.append(np.sum(~same))
    t = np.array(t); Rg = np.array(Rg); R2 = np.array(R2)
    dt = t[1] - t[0]
    N = len(chains[0]) * len(chains)
    win = t >= T_START_PS - 1e-6
    out = dict(system=system, n_frames=len(t), dt_ps=float(dt), t_end_ps=float(t[-1]), N_res=N)
    for name, y in (("Rg", Rg), ("R2", R2)):
        tau_full, z_full = acf_tau(y, dt)
        tau_win, z_win = acf_tau(y[win], dt)
        T_full = t[-1] - t[0]; T_win = t[win][-1] - t[win][0]
        h = win.sum() // 2
        out[name] = dict(mean_win=float(y[win].mean()), tau_full_ps=float(tau_full),
                         first_zero_full_ps=float(z_full), Neff_full=float(T_full / (2 * tau_full)),
                         tau_win_ps=float(tau_win), Neff_win=float(T_win / (2 * tau_win)),
                         half1=float(y[win][:h].mean()), half2=float(y[win][h:].mean()),
                         first_frame=float(y[0]), last_frame=float(y[-1]))
    ci = np.array(c_intra)[win] / N; ce = np.array(c_inter)[win] / N
    out["contacts_intra_per_res"] = dict(mean=float(ci.mean()), std=float(ci.std()))
    out["contacts_inter_per_res"] = dict(mean=float(ce.mean()), std=float(ce.std()))
    out["series"] = dict(t_ps=t.tolist(), Rg=Rg.tolist(), R2=R2.tolist(),
                         contacts_intra=np.array(c_intra).tolist())
    return out


if __name__ == "__main__":
    system = sys.argv[1]
    r = run(system)
    dest = Path(__file__).resolve().parent.parent / "resultados" / f"{system}_relax.json"
    dest.write_text(json.dumps(r, indent=1))
    for k in ("Rg", "R2"):
        d = r[k]
        print(f"{system} {k}: mean(2-10ns)={d['mean_win']:.3f} tau_full={d['tau_full_ps']/1000:.2f} ns "
              f"Neff_full={d['Neff_full']:.1f} | tau_win={d['tau_win_ps']/1000:.2f} ns Neff_win={d['Neff_win']:.1f} "
              f"| halves {d['half1']:.3f} / {d['half2']:.3f} | first frame {d['first_frame']:.3f}")
    print(f"{system} contacts/res: intra {r['contacts_intra_per_res']['mean']:.2f} ± {r['contacts_intra_per_res']['std']:.2f}"
          f"  inter {r['contacts_inter_per_res']['mean']:.2f} ± {r['contacts_inter_per_res']['std']:.2f}")
