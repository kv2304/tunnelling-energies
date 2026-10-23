"""
Tilted double well energy levels by finite difference.

[x] = nm, for all lengths
[E] = eV
[ψ] = nm^(-1/2)
"""

import os

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import curve_fit, linear_sum_assignment
from scipy.sparse import diags_array

HBAR2_OVER_2M = 0.0380998  # hbar^2 / 2m_e, eV nm^2
A = 1.0  # well width
V0 = 0.30  # barrier height

H_GRID = 0.010  # grid spacing
M = 2100  # number of grid points

WS = np.arange(1.0, 8.01, 0.5)  # well separations
V0S = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35])  # barrier heights
N_LMBDA = 21  # number of tilts per side
W_FIT = 2.0  # minimum fitted separation

W = 8.0  # well separation, untilted
W_B, LMBDA_B = 1.0, -0.03  # tilted well separation, tilt


def grid(h, M):
    """
    M interior points at spacing h, with the psi = 0 walls one h beyond each end.
    """
    L = (M + 1) * h
    return -L / 2 + h * np.arange(1, M + 1)


def hamiltonian(x, lmbda, w, a=A, V0=V0):
    """
    Build the finite-difference Hamiltonian for a tilted double well.

    Returns H (for eigenstates/eigenvalues) and V (to verify potential shape).
    """
    h = (x[-1] - x[0]) / (x.size - 1)
    if not np.allclose(np.diff(x), h):
        raise ValueError("Non-uniform grid spacing.")

    w = round(w / h) * h  # snap to grid
    a = round(a / h) * h

    r = np.round(2 * np.abs(x) / h) * h / 2  # |x| on half-steps
    in_well = (r >= w / 2) & (r < w / 2 + a)

    V = np.full(x.size, V0)
    V[in_well] = np.sign(x[in_well]) * lmbda / 2

    k = HBAR2_OVER_2M / h ** 2
    off = np.full(x.size - 1, -k)
    dia = 2 * k + V
    H = diags_array([off, dia, off], format="csr", offsets=[-1, 0, 1])  # type: ignore

    return H, V


def track(x, lmbdas, w, n_track=2, a=A, V0=V0):
    """
    Track the lowest n_track levels and states across a lmbda sweep.

    Returns E, psi (normalised on the grid) and ovl_min, the worst overlap matched.
    """
    E = np.empty((lmbdas.size, n_track))
    psi = np.empty((lmbdas.size, x.size, n_track))

    def lowest(lmbda):
        H = hamiltonian(x, lmbda, w, a, V0)[0]
        return eigh_tridiagonal(H.diagonal(0), H.diagonal(-1), select="i", select_range=(0, n_track - 1))

    prev = lowest(lmbdas[0])[1]
    top = np.abs(prev).argmax(axis=0)
    prev = prev * np.sign(prev[top, np.arange(n_track)])

    # match states step to step
    ovl_min = 1.0
    for i, lmbda in enumerate(lmbdas):
        Ei, vi = lowest(lmbda)

        ovl = np.abs(prev.T @ vi)  # overlaps
        keep = linear_sum_assignment(-ovl)[1]  # best match
        ovl_min = min(ovl_min, ovl[np.arange(n_track), keep].min())

        Ei, vi = Ei[keep], vi[:, keep]
        vi = vi * np.sign(np.einsum("ij,ij->j", prev, vi))  # fix sign
        E[i], psi[i], prev = Ei, vi, vi

    return E, psi, ovl_min


def splitting(x, w, a=A, V0=V0):
    """
    Tunnelling splitting 2t, the separation of the lowest two levels at lmbda = 0.
    """
    E = track(x, np.array([0.0]), w, 2, a, V0)[0][0]
    return E[1] - E[0]


def sweep(x, ws, n_lmbda=N_LMBDA, lmbda_max=12.0, n_track=2, a=A, V0=V0):
    """
    Sweep lmbda at each w, out to +- max|lmbda| / 2t.

    Returns lmbdas, E and two_t, one per w.
    """
    two_t = np.empty(ws.size)
    lmbdas = np.empty((ws.size, 2 * n_lmbda - 1))
    E = np.empty((ws.size, 2 * n_lmbda - 1, n_track))

    for i, w in enumerate(ws):
        two_t[i] = splitting(x, w, a, V0)  # scales the lmbda range

        half = np.linspace(0.0, lmbda_max * two_t[i], n_lmbda)
        lmbdas[i] = np.concatenate([-half[:0:-1], half])  # includes lmbda = 0
        E[i] = track(x, lmbdas[i], w, n_track, a, V0)[0]

    return lmbdas, E, two_t


def two_level_gap(lmbda, d, two_t):
    """
    Analytical two-level model of the gap.
    """
    return np.sqrt((d * lmbda) ** 2 + two_t ** 2)


def measure(x, run):
    """
    Run every check: box, nodes, tracking, d and 2t, kappa, float64 floor, convergence.

    Returns the results as a dict.
    """
    h = (x[-1] - x[0]) / (x.size - 1)
    lmbda0 = np.array([0.0])
    lmbdas, Esw, two_t, EV = run["lmbdas"], run["E"], run["two_t"], run["EV"]
    iw = int(np.argmin(np.abs(WS - W_FIT)))
    keep = WS >= W_FIT  # w used in the slope fits

    # box check
    n = np.arange(1, 7)
    box = []
    for hb, Mb in ((h, x.size), (h / 2, 2 * x.size)):
        Eb = track(grid(hb, Mb), lmbda0, W, n.size, A, 0.0)[0][0]
        box.append(Eb / (HBAR2_OVER_2M * (n * np.pi / ((Mb + 1) * hb)) ** 2) - 1)
    box_rel, box_ratio = box[0], np.mean(box[0] / box[1])

    # node check
    E8, psi8, _ = track(x, lmbda0, W, 8, A, V0)
    E8, psi8 = E8[0], psi8[0] / np.sqrt(h)  # normalise to unit integral
    nodes = [int((np.diff(np.sign(p[p != 0])) != 0).sum()) for p in psi8.T]

    # tracking check
    lmbda_tr = np.linspace(-0.05, 0.05, 41)  # through the crossing at W_B
    Etr, _, ovl = track(x, lmbda_tr, W_B, 2, A, V0)
    swaps = int(np.sum(np.diff(np.sign(Etr[:, 0] - Etr[:, 1])) != 0))

    # two-level check
    gap_iw = Esw[iw, :, 1] - Esw[iw, :, 0]
    (d_fit, two_t_fit), _ = curve_fit(two_level_gap, lmbdas[iw], gap_iw, p0=(1.0, two_t[iw]))
    mL = (x >= -(WS[iw] / 2 + A)) & (x < -WS[iw] / 2)
    mR = (x > WS[iw] / 2) & (x < WS[iw] / 2 + A)
    psi0 = track(x, lmbda0, WS[iw], 2, A, V0)[1][0]
    loc = (psi0[:, 0] + psi0[:, 1]) / np.sqrt(2)  # localised state
    d_psi = abs((loc[mL] ** 2).sum() - (loc[mR] ** 2).sum())

    # kappa check
    i0 = lmbdas.shape[1] // 2
    kappa = np.sqrt((V0 - Esw[:, i0, :].mean(axis=1)) / HBAR2_OVER_2M)
    slope, icept = np.polyfit(WS[keep], np.log(two_t[keep]), 1)
    two_t_V = EV[:, :, 1] - EV[:, :, 0]
    kappa_V = np.sqrt((V0S[:, None] - EV.mean(axis=2)) / HBAR2_OVER_2M)[:, keep]
    slope_V = np.array([np.polyfit(WS[keep], np.log(g[keep]), 1)[0] for g in two_t_V])
    kappa_V_worst = np.max(np.abs(slope_V + kappa_V.mean(axis=1)) / kappa_V.mean(axis=1))

    # floor check
    floor = np.finfo(float).eps * 4 * HBAR2_OVER_2M / h ** 2
    ws_floor = (11.0, 12.0, 13.0)
    m_wide = 2700  # wider box, fits w = 13
    xb = grid(h, m_wide)
    two_t_floor = [splitting(xb, wf, A, V0) for wf in ws_floor]

    # convergence check
    m_fine = 3 * (x.size + 1) - 1  # same box at h/3
    conv = []
    for hc, Mc in ((h, x.size), (h / 3, m_fine), (h, m_wide)):
        xc = grid(hc, Mc)
        splits = [splitting(xc, wc, A, V0) for wc in WS[keep]]
        conv.append(np.polyfit(WS[keep], np.log(splits), 1)[0])

    return dict(
        h=h, ws=WS, V0s=V0S, keep=keep, two_t=two_t,
        box_rel=box_rel, box_ratio=box_ratio, E8=E8, psi8=psi8, nodes=nodes,
        ovl_min=ovl, swaps=swaps,
        two_t_direct=two_t[iw], two_t_V=two_t_V, d_fit=d_fit, d_psi=d_psi, two_t_fit=two_t_fit,
        slope=slope, icept=icept,
        kappa=kappa[keep].mean(), slope_V=slope_V, kappa_V=kappa_V.mean(axis=1),
        kappa_V_worst=kappa_V_worst,
        floor=floor, ws_floor=ws_floor, two_t_floor=two_t_floor, conv=conv)


def figures(x, run, results, outdir="figures"):
    """
    Draw the figures from run and the results of measure, saved to outdir.
    """
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Bbox

    os.makedirs(outdir, exist_ok=True)
    plt.rcParams.update({
        "figure.figsize": (3.5, 2.6), "savefig.dpi": 300,
        "savefig.bbox": "tight", "font.family": "serif",
        "mathtext.fontset": "cm", "font.size": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "axes.linewidth": 0.8, "lines.linewidth": 1.4,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "in", "ytick.direction": "in",
        "legend.frameon": False, "axes.titlesize": 7.5, "axes.titlepad": 4.0,
    })
    COLOURS = ["#009E73", "#D55E00", "#CC79A7", "#0072B2"]
    PAIR = (4.9, 3.2)
    LEG = dict(frameon=False, columnspacing=1.1, handlelength=1.3, handletextpad=0.5, borderpad=0.0)
    h = results["h"]
    ws, two_t, lmbdas = results["ws"], results["two_t"], run["lmbdas"]
    iw = int(np.argmin(np.abs(ws - W_FIT)))
    W_fig = ws[iw]
    E = results["E8"]
    kappa = results["kappa"]

    # setup.svg
    W_A, lmbda_s = ws[0], lmbdas[0].min()
    two_t_A = float(two_t[0])
    Vd = hamiltonian(x, lmbda_s, W_A, A, V0)[1]
    arr = dict(arrowstyle="<->", lw=0.8, color="0.35", shrinkA=0, shrinkB=0)

    fig, ax = plt.subplots(figsize=(4.8, 3.2))

    ax.axvspan(-W_A / 2, W_A / 2, 0.0, 0.82, color="0.94", zorder=0)
    ax.axhline(0.0, color="0.7", lw=0.7, ls=":", zorder=0)
    for edge in (-W_A / 2 - A, -W_A / 2, W_A / 2, W_A / 2 + A):
        ax.axvline(edge, color="0.85", lw=0.6, zorder=0)
    ax.plot(x, Vd, color="0.25")
    ax.annotate("", (-W_A / 2, 0.345), (W_A / 2, 0.345), arrowprops=arr)
    ax.annotate(rf"$w={W_A:.0f}$ nm", (0.0, 0.365), ha="center", fontsize=7.5, color="0.25")
    ax.annotate("", (W_A / 2, 0.345), (W_A / 2 + A, 0.345), arrowprops=arr)
    ax.annotate(rf"$a={A:.0f}$ nm", (W_A / 2 + A + 0.12, 0.345), va="center", fontsize=7.5, color="0.25")
    ax.annotate("", (2.45, 0.0), (2.45, V0), arrowprops=arr)
    ax.annotate(rf"$V_0={V0}$ eV", (2.6, V0 / 2), va="center", fontsize=7.5, color="0.25")
    ax.annotate(r"$-\lambda/2$", (-W_A / 2 - A - 0.12, -lmbda_s / 2),
                ha="right", va="center", fontsize=7.5, color="0.25")
    ax.annotate(r"$+\lambda/2$", (W_A / 2 + A + 0.12, lmbda_s / 2),
                ha="left", va="center", fontsize=7.5, color="0.25")
    ax.set_xlim(-3.5, 4.4)
    ax.set_ylim(-0.155, 0.40)
    ax.set_xlabel(r"$x$ (nm)")
    ax.set_ylabel(r"$V$ (eV)")
    ax.set_title(
        rf"$\lambda={lmbda_s:.3f}$ eV, $|\lambda|/2t={abs(lmbda_s / two_t_A):.1f}$;  "
        rf"sampled at {x.size} points, $h={h}$ nm:" "\n"
        rf"{round(A / h)} per well, {round(1 / (kappa * h))} per $1/\kappa$")

    fig.savefig(f"{outdir}/setup.svg")

    # spectrum.svg
    Vdw = hamiltonian(x, 0.0, W, A, V0)[1]
    gap_neV = (E[1] - E[0]) * 1e9
    arr_i = dict(arrowstyle="<->", lw=0.7, color="0.35", shrinkA=0, shrinkB=0)

    fig, ax = plt.subplots(figsize=PAIR)

    ax.plot(x, Vdw, color="0.55", lw=1.0, label=r"$V(x)$")
    ax.axhline(V0, color="black", lw=0.6, ls=":", label=r"$V_0$")
    for i in range(4):
        ax.plot(x, np.where(Vdw < E[i], E[i], np.nan), lw=1.3, color=COLOURS[i], label=rf"$E_{i + 1}$")
    ax.annotate("1, 2", (11.0, E[0]), fontsize=7, va="center", color="0.25")
    ax.annotate("3, 4", (11.0, 0.5 * (E[2] + E[3])), fontsize=7, va="center", color="0.25")
    ax.set_xlim(-12.6, 12.6)
    ax.set_ylim(-0.03, 0.47)
    ax.set_xlabel(r"$x$ (nm)")
    ax.set_ylabel(r"$E$ (eV)")
    ax.set_title(rf"$w={W:.0f}$ nm, $\lambda=0$: levels drawn only where $E>V(x)$")

    axi = ax.inset_axes((0.06, 0.13, 0.21, 0.26))
    axi.set_facecolor("white")
    for i in (0, 1):
        axi.axhline((E[i] - E[0]) * 1e9, color=COLOURS[i], lw=1.3)
    axi.annotate("", (0.42, 0.0), (0.42, gap_neV), xycoords=("axes fraction", "data"), arrowprops=arr_i)
    axi.annotate(f"{gap_neV:.2f} neV", (0.5, gap_neV / 2), xycoords=("axes fraction", "data"),
                 fontsize=6, va="center", color="0.25")
    axi.set_ylim(-0.7 * gap_neV, 1.7 * gap_neV)
    axi.set_xticks([])
    axi.set_yticks([])
    axi.set_title(r"$E_1, E_2$ magnified", fontsize=6.5)
    ax.legend(loc="upper center", ncol=3, **LEG)

    fig.savefig(f"{outdir}/spectrum.svg", bbox_inches=Bbox([[0, 0], PAIR]))

    # eigenstates.svg
    psi, nodes = results["psi8"], results["nodes"]

    fig, ax = plt.subplots(2, 2, figsize=PAIR)
    ax = ax.ravel()

    for i in range(4):
        ax[i].axhline(0.0, color="0.8", lw=0.6, zorder=0)
        for edge in (-W / 2 - A, -W / 2, W / 2, W / 2 + A):
            ax[i].axvline(edge, color="0.91", lw=0.6, zorder=0)
        ax[i].plot(x, psi[:, i], color=COLOURS[i], lw=1.1)
        ax[i].margins(y=0.20)
        ax[i].set_xlim(x[0], x[-1])
        ax[i].set_xlabel(r"$x$ (nm)", fontsize=7)
        node_s = "node" if nodes[i] == 1 else "nodes"
        ax[i].set_title(rf"$n={i + 1}$,  {nodes[i]} {node_s},  $E={E[i]:.4f}$ eV", loc="left", fontsize=6.3)
        ax[i].tick_params(labelsize=6)
    for i in (0, 2):
        ax[i].set_ylabel(r"$\psi$ (nm$^{-1/2}$)", fontsize=7)

    fig.suptitle(rf"$w={W:.0f}$ nm,  $\lambda=0$,  $\Delta E_{{i,j}} \neq 0$", fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 1.0))
    fig.align_ylabels()

    fig.savefig(f"{outdir}/eigenstates.svg", bbox_inches=Bbox([[0, 0], PAIR]))

    # tilted_wells.svg
    Eb, psib, _ = track(x, np.array([LMBDA_B]), W_B, 2, A, V0)
    Eb, pb = Eb[0], (psib[0] / np.sqrt(h)) ** 2
    Vb = hamiltonian(x, LMBDA_B, W_B, A, V0)[1]
    tB = float(splitting(x, W_B))
    peak = 0.042  # plotted peak height
    scale = peak / pb.max()

    fig, ax = plt.subplots(figsize=(4.2, 3.1))

    ax.axvspan(-W_B / 2, W_B / 2, color="0.93", zorder=0)
    ax.plot(x, Vb, color="0.55", lw=1.0, label=r"$V(x)$")
    for i in range(2):
        ax.axhline(Eb[i], color="0.88", lw=0.5, zorder=0)
        ax.plot(x, Eb[i] + scale * pb[:, i], color=COLOURS[i], lw=1.2, label=rf"$|\psi_{i + 1}|^2$")
    ax.annotate(r"$-\lambda/2$", (-1.78, -LMBDA_B / 2), fontsize=7, ha="center", va="center", color="0.25")
    ax.annotate(r"$+\lambda/2$", (1.78, LMBDA_B / 2), fontsize=7, ha="center", va="center", color="0.25")
    ax.set_xlim(-2.95, 2.95)
    ax.set_ylim(-0.075, 0.198)
    ax.set_xlabel(r"$x$ (nm)")
    ax.set_ylabel(r"$E$ (eV)")
    ax.set_title(
        rf"$w={W_B:.0f}$ nm,  $\lambda={LMBDA_B}$ eV, $|\lambda|/2t={abs(LMBDA_B / tB):.1f}$"
        + 4 * " " + r"$|\psi|^2$ scale arbitrary")
    ax.legend(loc="lower left", ncol=1, **LEG)

    fig.savefig(f"{outdir}/tilted_wells.svg")

    # avoided_crossing.svg
    lf, d_fit, two_t_fit = lmbdas[iw], results["d_fit"], results["two_t_fit"]
    Ef, psif, _ = track(x, lf, W_fig, 2, A, V0)
    mL = (x >= -(W_fig / 2 + A)) & (x < -W_fig / 2)
    mR = (x > W_fig / 2) & (x < W_fig / 2 + A)
    pl = (psif[:, mL, :] ** 2).sum(axis=1)
    pr = (psif[:, mR, :] ** 2).sum(axis=1)
    frac = pl / (pl + pr)
    mid, fit_gap = Ef.mean(axis=1), two_level_gap(lf, d_fit, two_t_fit)
    lo, hi = (Ef * 1e3).min(), (Ef * 1e3).max()

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    for i in range(2):
        ax0.plot(lf * 1e3, Ef[:, i] * 1e3, color=COLOURS[i], label=rf"$E_{i + 1}$")
    for sgn in (+1, -1):
        label = "two-level fit" if sgn > 0 else None
        ax0.plot(lf * 1e3, (mid + sgn * fit_gap / 2) * 1e3, ls="--", lw=0.8, color="black", label=label)
    ax0.set_ylim(lo - 0.12 * (hi - lo), hi + 0.32 * (hi - lo))
    ax0.set_xlabel(r"$\lambda$ (meV)")
    ax0.set_ylabel(r"$E$ (meV)")
    ax0.set_title(rf"$w={W_fig:.1f}$ nm,  min gap $2t={two_t_fit * 1e3:.3f}$ meV,  fitted $d={d_fit:.3f}$")
    ax0.legend(loc="upper center", ncol=3, **LEG)

    for i in range(2):
        ax1.plot(lf * 1e3, frac[:, i], color=COLOURS[i], label=rf"state {i + 1}")
    ax1.axhline(0.5, color="0.85", lw=0.6, ls=":")
    ax1.set_ylim(-0.05, 1.38)
    ax1.set_yticks([0.0, 0.5, 1.0])
    ax1.set_xlabel(r"$\lambda$ (meV)")
    ax1.set_ylabel("fraction of well probability on the left")
    ax1.set_title(r"$\lambda>0$ lowers the left well, so each state changes side")
    ax1.legend(loc="upper center", ncol=2, **LEG)

    fig.savefig(f"{outdir}/avoided_crossing.svg")

    # log_gap.svg
    keep = results["keep"]
    slope, icept = results["slope"], results["icept"]
    V0s, two_t_V = results["V0s"], results["two_t_V"]
    blues = plt.get_cmap("Blues")(np.linspace(0.45, 0.97, V0s.size))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    ax0.plot(ws[~keep], np.log(two_t[~keep]), "o", ms=3.5, mfc="none", mec="0.6",
             label=rf"$w<{W_FIT:g}$ nm, excluded")
    ax0.plot(ws[keep], np.log(two_t[keep]), "o", ms=3.5, color=COLOURS[3], label="data")
    ax0.plot(ws, slope * ws + icept, ls="--", lw=0.9, color="black", label=rf"fit, $w\geq{W_FIT:g}$ nm")
    ax0.set_xlim(0.4, 8.6)
    ax0.set_xlabel(r"$w$ (nm)")
    ax0.set_ylabel(r"$\ln(2t\,/\,\mathrm{eV})$")
    ax0.set_title(rf"slope ${slope:.5f}$   vs   $-\kappa = {-kappa:.5f}$ nm$^{{-1}}$   "
                  rf"({abs(slope + kappa) / kappa:.3%})")
    ax0.legend(loc="upper right", **LEG)

    for j, v in enumerate(V0s):
        ax1.plot(ws[keep], np.log(two_t_V[j][keep]), "o-", ms=2.5, lw=1.0, color=blues[j])
        ax1.annotate(rf"${v:.2f}$", (8.25, np.log(two_t_V[j][keep][-1])),
                     fontsize=6.5, va="center", color="0.25")
    ax1.set_xlim(1.7, 8.9)
    ax1.set_xlabel(r"$w$ (nm)")
    ax1.set_ylabel(r"$\ln(2t\,/\,\mathrm{eV})$")
    ax1.set_title("labelled by $V_0$ (eV)\n"
                  rf"each slope matches its own $\kappa$ to $\leq{100 * results['kappa_V_worst']:.1f}\%$")

    fig.tight_layout()
    fig.savefig(f"{outdir}/log_gap.svg")


if __name__ == "__main__":
    x = grid(H_GRID, M)
    lmbda0 = np.array([0.0])

    lmbdas, Esw, two_t = sweep(x, WS)
    EV = np.array([[track(x, lmbda0, w, 2, A, v)[0][0] for w in WS] for v in V0S])
    run = dict(lmbdas=lmbdas, E=Esw, two_t=two_t, EV=EV)

    results = measure(x, run)

    rel, ratio = results["box_rel"], results["box_ratio"]
    slope, kappa, floor = results["slope"], results["kappa"], results["floor"]
    conv = results["conv"]
    kappa_dev = abs(slope + kappa) / kappa
    kappa_V_worst = results["kappa_V_worst"]
    slope_range = max(conv) - min(conv)

    print(f"grid  L = {(M + 1) * H_GRID:.2f} nm,  h = {H_GRID} nm,  M = {M}\n")
    print(f"{'check':<16}{'quantity':<27}measured | expected")
    print(f"box energies    {f'n = 1..{rel.size} vs analytic':<27}{np.abs(rel).max():.1e} | O(h^2)")
    print(f"box, O(h^2)     error ratio on halving h   {ratio:.3f} | 4")
    nodes = results["nodes"]
    print(f"node theorem    {f'nodes for n = 1..{len(nodes)}':<27}{nodes} | "
          f"n-1 for state n")
    print(f"tracking        crossings, worst overlap   {results['swaps']} | 0,  "
          f"{results['ovl_min']:.4f} | >0.9")
    print(f"two-level fit   d fit | from |psi|^2       {results['d_fit']:.4f} | {results['d_psi']:.4f}")
    print(f"                2t fit | direct            "
          f"{results['two_t_fit']:.4e} | {results['two_t_direct']:.4e}")
    print(f"log-gap slope   fit | kappa                {slope:+.5f} | {-kappa:+.5f}   ({kappa_dev:.3%})")
    print(f"vs height       {f'worst of {V0S.size} barrier heights':<27}{kappa_V_worst:.2%}")
    print(f"precision       floor | smallest swept     {floor:.1e} | {two_t.min():.2e} eV")
    two_t_floor = results["two_t_floor"]
    ws_fl = ", ".join(f"{w:g}" for w in results["ws_floor"])
    print(f"                {f'gaps at w = {ws_fl} nm':<27}"
          f"{', '.join(f'{g:.1e}' for g in two_t_floor)} | last below the floor")
    print(f"convergence     slope range, h/3 and box   {slope_range:.1e} nm^-1   "
          f"({slope_range / abs(np.mean(conv)):.3%})")

    figures(x, run, results)

