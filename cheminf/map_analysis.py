#!/usr/bin/env python3
"""Analyse the xtb (phi, psi) scan: minima, smoothness, Boltzmann populations, and comparison
with the MD (phi, psi) distributions. Writes ../resultados/map_summary.json and a figure.

    python3 map_analysis.py [scan_10deg.dat]
"""
import sys, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
KT = 0.0019872 * 303.15          # kcal/mol
HARTREE = 627.509


def load(f):
    rows = [l.split() for l in open(f) if len(l.split()) >= 3 and l.split()[0] != "DONE"]
    if rows and len(rows[0]) >= 6:                   # validated scan (validate_frames.py): keep valid == 1
        nbad = sum(1 for r in rows if r[5] != "1")
        print(f"validated scan: {nbad} of {len(rows)} frames excluded (connectivity / ring / drift)")
        rows = [r for r in rows if r[5] == "1"]
    a = np.array([[float(x) for x in r[:5]] for r in rows if len(r) >= 5])
    dphi = (a[:, 3] - a[:, 0] + 180) % 360 - 180; dpsi = (a[:, 4] - a[:, 1] + 180) % 360 - 180
    ok = (np.abs(dphi) < 5) & (np.abs(dpsi) < 5)
    print(f"torsion drift > 5 deg at {(~ok).sum()} points (dropped); max |drift| {np.abs(np.r_[dphi, dpsi]).max():.1f} deg")
    a = a[ok]
    phi, psi, E = a[:, 0], a[:, 1], (a[:, 2] - a[:, 2].min()) * HARTREE
    return phi, psi, E


def grid(phi, psi, E, step=10):
    P = np.arange(-180, 180, step); S = np.arange(-180, 180, step)
    G = np.full((len(P), len(S)), np.nan)
    for f, s, e in zip(phi, psi, E):
        G[int(round((f + 180) / step)) % len(P), int(round((s + 180) / step)) % len(S)] = e
    return P, S, G


def main():
    f = sys.argv[1] if len(sys.argv) > 1 else HERE / "scan_10deg.dat"
    phi, psi, E = load(f)
    P, S, G = grid(phi, psi, E)
    done = np.isfinite(G)
    print(f"{done.sum()} / {G.size} grid points; E range 0 – {np.nanmax(G):.1f} kcal/mol")
    # smoothness: rms difference between neighbouring cells (rotamer hops show as spikes)
    d1 = np.abs(np.diff(G, axis=0)); d2 = np.abs(np.diff(G, axis=1))
    low = G < 6
    print(f"neighbour |dE|: median {np.nanmedian(np.concatenate([d1.ravel(), d2.ravel()])):.2f}, "
          f"90% {np.nanpercentile(np.concatenate([d1.ravel(), d2.ravel()]), 90):.2f} kcal/mol; "
          f"cells with E < 6 kcal/mol: {np.nansum(low)}")
    # local minima (8-neighbour, periodic)
    mins = []
    for i in range(len(P)):
        for j in range(len(S)):
            if not done[i, j]: continue
            nb = [G[(i + a) % len(P), (j + b) % len(S)] for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
            if np.all(np.isfinite(nb)) and G[i, j] < min(nb):
                mins.append((G[i, j], P[i], S[j]))
    mins.sort()
    print("local minima (E kcal/mol, phi, psi):", [(round(e, 1), int(p), int(s)) for e, p, s in mins[:8]])
    # Boltzmann populations
    w = np.where(done, np.exp(-np.nan_to_num(G, nan=1e9) / KT), 0.0); w /= w.sum()
    mphi = np.degrees(np.angle(np.sum(w * np.exp(1j * np.radians(P))[:, None])))
    mpsi = np.degrees(np.angle(np.sum(w * np.exp(1j * np.radians(S))[None, :])))
    wsorted = np.sort(w.ravel())[::-1]
    print(f"Boltzmann (303 K): mean phi {mphi:.0f}, psi {mpsi:.0f}; cells holding 50 % / 90 % of weight: "
          f"{np.searchsorted(np.cumsum(wsorted), 0.5) + 1} / {np.searchsorted(np.cumsum(wsorted), 0.9) + 1}")
    # population in basins: syn (phi 60..140, psi -180..-90), and the rest
    syn = (P[:, None] >= 60) & (P[:, None] <= 140) & (S[None, :] >= -180) & (S[None, :] <= -90)
    print(f"weight in syn basin (phi 60–140, psi −180…−90): {w[syn].sum():.3f}")
    # compare with MD histograms
    out = dict(points=int(done.sum()), minima=[(float(e), int(p), int(s)) for e, p, s in mins[:8]],
               mean_phi=float(mphi), mean_psi=float(mpsi), w_syn=float(w[syn].sum()))
    for sysn in ("SC6", "SC18", "DC6"):
        fn = HERE / f"phipsi_{sysn}.npy"
        if not fn.exists(): continue
        pp = np.load(fn)
        H, _, _ = np.histogram2d(pp[:, 0], pp[:, 1], bins=[np.arange(-185, 185, 10), np.arange(-185, 185, 10)])
        H = H / H.sum()
        # overlap coefficient sum(min) and MD-weighted mean energy
        ov = np.sum(np.minimum(H, w)) if H.shape == w.shape else float("nan")
        Emd = np.nansum(H * np.nan_to_num(G, nan=0.0)) / np.sum(H[done])
        print(f"MD {sysn}: overlap with xtb Boltzmann = {ov:.2f}; MD-weighted xtb energy = {Emd:.2f} kcal/mol; "
              f"MD circular mean phi {np.degrees(np.angle(np.mean(np.exp(1j*np.radians(pp[:,0]))))):.0f} psi {np.degrees(np.angle(np.mean(np.exp(1j*np.radians(pp[:,1]))))):.0f}")
        out[f"overlap_{sysn}"] = float(ov)
    (HERE.parent / "resultados" / "map_summary.json").write_text(json.dumps(out, indent=1))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4.2))
        im = ax.imshow(G.T, origin="lower", extent=[-185, 175, -185, 175], vmin=0, vmax=12, cmap="viridis", aspect="equal")
        cs = ax.contour(P, S, G.T, levels=[1, 2, 3, 5, 8], colors="w", linewidths=0.6)
        for sysn, c in (("SC18", "orange"), ("DC6", "red")):
            fn = HERE / f"phipsi_{sysn}.npy"
            if fn.exists():
                pp = np.load(fn); ax.scatter(pp[::20, 0], pp[::20, 1], s=2, c=c, alpha=0.4, label=f"MD {sysn}")
        ax.set_xlabel("φ  O5′–C1′–O4–C4 (°)"); ax.set_ylabel("ψ  C1′–O4–C4–C5 (°)"); ax.legend(loc="upper left", fontsize=7)
        plt.colorbar(im, ax=ax, label="E (kcal/mol), GFN2-xTB/ALPB")
        fig.tight_layout(); fig.savefig(HERE.parent / "figuras" / "map_phipsi_xtb.png", dpi=200)
        print("figure: figuras/map_phipsi_xtb.png")
    except Exception as e:
        print("no figure:", e)


if __name__ == "__main__":
    main()
