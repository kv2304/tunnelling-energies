"""
Tilted double well energy levels by finite difference. 

[E] = eV
[ψ] = nm^(-1/2)
"""

import os

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import linear_sum_assignment
from scipy.optimize import curve_fit
from scipy.sparse import diags_array

HBAR2_OVER_2M = 0.0380998   # hbar^2 / 2m_e, eV nm^2
A = 1.0                     # well width, nm
V0 = 0.30                   # well depth, eV
W_FIT = 2.0                 # nm 
M = 2100                    # interior points
W, W_B, LAMDA_B = 8.0, 1.0, -0.03


def hamiltonian(x, lamda, w, a=A, V0=V0):
    """
    Build the finite-difference Hamiltonian for a tilted double well.

    Discretises the TISE on grid x, giving a tridiagonal H with kinetic
    coupling -k off-diagonal and (2k + V_i) on-diagonal. V is ±lamda/2 inside
    two wells of width a separated by gap w and V0 elsewhere.

    Returns H (for eigenstates/eigenvalues) and V (to verify potential shape).
    """
    h = (x[-1] - x[0]) / (x.size - 1)
    if not np.allclose(np.diff(x), h):
        raise ValueError("Non-uniform grid spacing.")

    w = round(w / h) * h
    a = round(a / h) * h

    r = np.round(2 * np.abs(x) / h) * h / 2
    in_well = (r >= w / 2) & (r < w / 2 + a)

    V = np.full(x.size, V0)
    V[in_well] = np.sign(x[in_well]) * lamda / 2

    # tridiagonal hamiltonian matrix
    k = HBAR2_OVER_2M / h**2  #eV
    off = np.full(x.size - 1, -k)  
    dia = 2 * k + V  
    H = diags_array([off, dia, off], format="csr", offsets=[-1, 0, 1])  # type: ignore

    return H, V


def track(x, lamdas, w, n_track=2, a=A, V0=V0):
    """
    Track the lowest n_track energy levels and eigenstates across a lamda sweep.

    Builds H at each lamda and diagonalises it, matching each new state to
    the last by overlap and fixing its sign to stay continuous.

    Returns E, psi (tracked) and ovl_min, the worst overlap matched.
    """
    E = np.empty((lamdas.size, n_track))
    psi = np.empty((lamdas.size, x.size, n_track))

    H = hamiltonian(x, lamdas[0], w, a, V0)[0]
    prev = eigh_tridiagonal(H.diagonal(0), H.diagonal(-1), select="i", select_range=(0, n_track - 1))[1]
    top = np.abs(prev).argmax(axis=0)
    prev = prev * np.sign(prev[top, np.arange(n_track)])

    # sequential by nature - each lambda continues the eigenvectors of the last
    ovl_min = 1.0
    for i, lamda in enumerate(lamdas):
        H = hamiltonian(x, lamda, w, a, V0)[0]
        Ei, vi = eigh_tridiagonal(H.diagonal(0), H.diagonal(-1), select="i", select_range=(0, n_track - 1))

        ovl = np.abs(prev.T @ vi)                   # |<prev|new>|: sign gone
        keep = linear_sum_assignment(-ovl)[1]       # one partner per state
        ovl_min = min(ovl_min, ovl[np.arange(n_track), keep].min())

        Ei, vi = Ei[keep], vi[:, keep]
        vi = vi * np.sign(np.einsum("ij,ij->j", prev, vi))   # keep sign smooth
        E[i], psi[i], prev = Ei, vi, vi

    return E, psi, ovl_min

def sweep(x, ws, n_lam=21, span=12.0, n_track=2, a=A, V0=V0):
    """
    Sweep lamda over a range of well separations, w, tracking n_track levels.

    For each w, first tracks at lamda=0 to get 2t, the pure tunnelling
    gap. The lamda grid is then scaled by 2t*span, to account for t dependence
    on w.

    Returns lamdas, E (tracked energies), and two_t, one per w.
    """
    two_t = np.empty(ws.size)
    lamdas = np.empty((ws.size, 2 * n_lam - 1))
    E = np.empty((ws.size, 2 * n_lam - 1, n_track))

    for i, w in enumerate(ws):
        gap0 = track(x, np.array([0.0]), w, n_track, a, V0)[0][0]
        two_t[i] = gap0[1] - gap0[0]        # sets the lambda scale below

        half = np.linspace(0.0, span * two_t[i], n_lam)
        lamdas[i] = np.concatenate([-half[:0:-1], half])   # contains 0 exactly
        E[i] = track(x, lamdas[i], w, n_track, a, V0)[0]

    return lamdas, E, two_t


def two_level_gap(lamda, d, two_t):
    """The 2x2 model in the localised basis: d lamda the energy difference
    the tilt opens between the wells, two_t the gap at lamda = 0."""
    return np.sqrt((d * lamda) ** 2 + two_t ** 2)


def measure(x, run, a=A, V0=V0) -> dict:
    """
    Computes bound states, orders states, and performs independent
    cross-checks of d and kappa (fit vs direct, plus grid convergence),
    returning the measured results as a dictionary.
    """
    h =(x[-1] - x[0]) / (x.size - 1)
    lamda0 = np.array([0.0])
    ws, V0s = run["ws"], run["V0s"]
    lamdas, Esw, two_t, EV = run["lamdas"], run["E"], run["two_t"], run["EV"]

    # the bound states themselves, and the node count as an independent check
    E8, psi8, _ = track(x, lamda0, W, 8, a, V0)
    E8, psi8 = E8[0], psi8[0] / np.sqrt(h)      # 2-norm -> unit integral
    nodes = [int((np.diff(np.sign(p[p != 0])) != 0).sum()) for p in psi8.T]

    # ordering and continuity of the tracked states through a crossing
    lamda_tr = np.linspace(-0.05, 0.05, 41)
    Etr, _, ovl = track(x, lamda_tr, W_B, 2, a, V0)
    swaps = int(np.sum(np.diff(np.sign(Etr[:, 0] - Etr[:, 1])) != 0))

    i0, iw = lamdas.shape[1] // 2, int(np.argmin(np.abs(ws - W_FIT)))
    # below w = 2 nm the wells are coupled strongly enough that the local slope has not yet reached -kappa
    keep = ws >= W_FIT
    kappa = np.sqrt((V0 - Esw[:, i0, :].mean(axis=1)) / HBAR2_OVER_2M)

    # tilt and coupling, two ways each: fitted, and read off the wavefunction
    (d_fit, t2_fit), _ = curve_fit(two_level_gap, lamdas[iw], Esw[iw, :, 1] - Esw[iw, :, 0], p0=(1.0, two_t[iw]))
    mL = (x >= -(ws[iw] / 2 + a)) & (x < -ws[iw] / 2)
    mR = (x > ws[iw] / 2) & (x < ws[iw] / 2 + a)
    psi0 = track(x, lamda0, ws[iw], 2, a, V0)[1][0]
    loc = (psi0[:, 0] + psi0[:, 1]) / np.sqrt(2)   # localised in one well
    d_psi = abs((loc[mL] ** 2).sum() - (loc[mR] ** 2).sum())

    # kappa, two ways: the log-gap slope, and sqrt((V0 - E)/(hbar^2/2m))
    (slope, icept), cov = np.polyfit(ws[keep], np.log(two_t[keep]), 1, cov=True)
    gV = EV[:, :, 1] - EV[:, :, 0]
    kV = np.sqrt((V0s[:, None] - EV.mean(axis=2)) / HBAR2_OVER_2M)[:, keep]
    sV = np.array([np.polyfit(ws[keep], np.log(g[keep]), 1)[0] for g in gV])

    # where float64 runs out, and whether the slope survives refinement
    floor = np.finfo(float).eps * 4 * HBAR2_OVER_2M / h**2
    M_WIDE = 2704                       # a wider box, and the same box at h/3
    M_FINE = 3 * (x.size + 1) - 1
    L_grid, L_wide = (x.size + 1) * h, (M_WIDE + 1) * h
    xb = -L_wide / 2 + h * np.arange(1, M_WIDE + 1)
    gap_fl = [np.diff(track(xb, lamda0, wf, 2, a, V0)[0][0])[0] for wf in (11.0, 12.0, 13.0)]

    conv = []
    for Lc, Mc in ((L_grid, x.size), (L_grid, M_FINE), (L_wide, M_WIDE)):
        hc = Lc / (Mc + 1)
        xc = -Lc / 2 + hc * np.arange(1, Mc + 1)
        conv.append(np.polyfit(ws[keep], [np.log(np.diff(track(xc, lamda0, wc, 2, a, V0)[0][0])[0]) for wc in ws[keep]], 1)[0])

    return dict(
        A=a, W=W, V0=V0, h=h, ws=ws, V0s=V0s, keep=keep, two_t=two_t,
        E8=E8, psi8=psi8, nodes=nodes,
        ovl=ovl, swaps=swaps,
        two_t_fig=two_t[iw], gV=gV, d_fit=d_fit, d_psi=d_psi, t2_fit=t2_fit,
        slope=slope, icept=icept, slope_se=np.sqrt(cov[0, 0]),
        k_ind=kappa[keep].mean(), sV=sV, kV=kV.mean(axis=1),
        floor=floor, gap_fl=gap_fl, conv=conv)


def figures(x, run, results, outdir="figures"):
    """
    Generate figures from the results dict of measure, computing the
    potentials and states drawn behind them from x and run.
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
    A, W, V0, h = results["A"], results["W"], results["V0"], results["h"]
    ws, two_t, lamdas = results["ws"], results["two_t"], run["lamdas"]
    iw = int(np.argmin(np.abs(ws - W_FIT)))
    W_fig = ws[iw]
    E = results["E8"]

    # setup.svg: the double wells at the true first point of a sweep
    W_A, lam_s = ws[0], lamdas[0].min()
    two_t_A = float(two_t[0])       # 2t at w_first
    Vd = hamiltonian(x, lam_s, W_A, A, V0)[1]
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
    ax.annotate(r"$-\lambda/2$", (-W_A / 2 - A - 0.12, -lam_s / 2), ha="right", va="center", fontsize=7.5, color="0.25")
    ax.annotate(r"$+\lambda/2$", (W_A / 2 + A + 0.12, lam_s / 2), ha="left", va="center", fontsize=7.5, color="0.25")
    ax.set_xlim(-3.5, 4.4)
    ax.set_ylim(-0.155, 0.40)
    ax.set_xlabel(r"$x$ (nm)")
    ax.set_ylabel(r"$V$ (eV)")
    ax.set_title(rf"$\lambda={lam_s:.3f}$ eV, $|\lambda|/2t={abs(lam_s / two_t_A):.1f}$;  sampled at {x.size} points, $h={h}$ nm:" "\n" rf"{round(A / h)} per well, {round(0.46 / h)} per $1/\kappa$")

    fig.savefig(f"{outdir}/setup.svg")

    # spectrum.svg: which energies the setup allows, with the pair's split inset
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
    axi.annotate(f"{gap_neV:.2f} neV", (0.5, gap_neV / 2), fontsize=6, xycoords=("axes fraction", "data"), va="center", color="0.25")
    axi.set_ylim(-0.7 * gap_neV, 1.7 * gap_neV)
    axi.set_xticks([])
    axi.set_yticks([])
    axi.set_title(r"$E_1, E_2$ magnified", fontsize=6.5)
    ax.legend(loc="upper center", ncol=3, **LEG)

    fig.savefig(f"{outdir}/spectrum.svg", bbox_inches=Bbox([[0, 0], PAIR]))

    # eigenstates.svg: psi_n for the lowest four, node count visible by eye
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
        ax[i].set_title(rf"$n={i + 1}$,  {nodes[i]} node" + ("s" if nodes[i] != 1 else "") + rf",  $E={E[i]:.4f}$ eV", loc="left", fontsize=6.3)
        ax[i].tick_params(labelsize=6)
    for i in (0, 2):
        ax[i].set_ylabel(r"$\psi$ (nm$^{-1/2}$)", fontsize=7)

    fig.suptitle(rf"$w={W:.0f}$ nm,  $\lambda=0$,  $\Delta E_{{i,j}} \neq 0$", fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 1.0))
    fig.align_ylabels()

    fig.savefig(f"{outdir}/eigenstates.svg", bbox_inches=Bbox([[0, 0], PAIR]))

    # tilted_wells.svg: |psi|^2 on baselines at E_n, once the tilt localises them
    Eb, psib, _ = track(x, np.array([LAMDA_B]), W_B, 2, A, V0)
    Eb, pb = Eb[0], (psib[0] / np.sqrt(h)) ** 2
    Vb = hamiltonian(x, LAMDA_B, W_B, A, V0)[1]
    tB = float(np.diff(track(x, np.array([0.0]), W_B, 2, A, V0)[0][0])[0])
    scale = 0.042 / pb.max()

    fig, ax = plt.subplots(figsize=(4.2, 3.1))

    ax.axvspan(-W_B / 2, W_B / 2, color="0.93", zorder=0)
    ax.plot(x, Vb, color="0.55", lw=1.0, label=r"$V(x)$")
    for i in range(2):
        ax.axhline(Eb[i], color="0.88", lw=0.5, zorder=0)
        ax.plot(x, Eb[i] + scale * pb[:, i], color=COLOURS[i], lw=1.2, label=rf"$|\psi_{i + 1}|^2$")
    ax.annotate(r"$-\lambda/2$", (-1.78, -LAMDA_B / 2), fontsize=7, ha="center", va="center", color="0.25")
    ax.annotate(r"$+\lambda/2$", (1.78, LAMDA_B / 2), fontsize=7, ha="center", va="center", color="0.25")
    ax.set_xlim(-2.95, 2.95)
    ax.set_ylim(-0.075, 0.198)
    ax.set_xlabel(r"$x$ (nm)")
    ax.set_ylabel(r"$E$ (eV)")
    ax.set_title(rf"$w={W_B:.0f}$ nm,  $\lambda={LAMDA_B}$ eV, $|\lambda|/2t={abs(LAMDA_B / tB):.1f}$" + 4 * " " + r"$|\psi|^2$ scale arbitrary")
    ax.legend(loc="lower left", ncol=1, **LEG)

    fig.savefig(f"{outdir}/tilted_wells.svg")

    # avoided_crossing.svg: the crossing, and the character exchange with it
    lf, d_fit, t2_fit = lamdas[iw], results["d_fit"], results["t2_fit"]
    Ef, psif, _ = track(x, lf, W_fig, 2, A, V0)
    mL = (x >= -(W_fig / 2 + A)) & (x < -W_fig / 2)
    mR = (x > W_fig / 2) & (x < W_fig / 2 + A)
    pl = (psif[:, mL, :] ** 2).sum(axis=1)
    pr = (psif[:, mR, :] ** 2).sum(axis=1)
    frac = pl / (pl + pr)
    mid, fit_gap = Ef.mean(axis=1), two_level_gap(lf, d_fit, t2_fit)
    lo, hi = (Ef * 1e3).min(), (Ef * 1e3).max()

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    for i in range(2):
        ax0.plot(lf * 1e3, Ef[:, i] * 1e3, color=COLOURS[i], label=rf"$E_{i + 1}$")
    for sgn in (+1, -1):
        ax0.plot(lf * 1e3, (mid + sgn * fit_gap / 2) * 1e3, ls="--", lw=0.8, color="black", label="two-level fit" if sgn > 0 else None)
    ax0.set_ylim(lo - 0.12 * (hi - lo), hi + 0.32 * (hi - lo))
    ax0.set_xlabel(r"$\lambda$ (meV)")
    ax0.set_ylabel(r"$E$ (meV)")
    ax0.set_title(rf"$w={W_fig:.1f}$ nm,  min gap $2t={t2_fit * 1e3:.3f}$ meV,  fitted $d={d_fit:.3f}$")
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

    # log_gap.svg: L one depth against its own kappa, R six depths each against theirs
    keep = results["keep"]
    slope, icept, k_ind = results["slope"], results["icept"], results["k_ind"]
    V0s, gV = results["V0s"], results["gV"]
    blues = plt.get_cmap("Blues")(np.linspace(0.45, 0.97, V0s.size))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    ax0.plot(ws[~keep], np.log(two_t[~keep]), "o", ms=3.5, mfc="none", mec="0.6", label=r"$w<2$ nm, excluded")
    ax0.plot(ws[keep], np.log(two_t[keep]), "o", ms=3.5, color=COLOURS[3], label="data")
    ax0.plot(ws, slope * ws + icept, ls="--", lw=0.9, color="black", label=r"fit, $w\geq2$ nm")
    ax0.set_xlim(0.4, 8.6)
    ax0.set_xlabel(r"$w$ (nm)")
    ax0.set_ylabel(r"$\ln(2t\,/\,\mathrm{eV})$")
    ax0.set_title(rf"slope ${slope:.5f}$   vs   $-\kappa = {-k_ind:.5f}$ nm$^{{-1}}$   ({abs(slope + k_ind) / k_ind:.3%})")
    ax0.legend(loc="upper right", **LEG)

    for j, v in enumerate(V0s):
        ax1.plot(ws[keep], np.log(gV[j][keep]), "o-", ms=2.5, lw=1.0, color=blues[j])
        ax1.annotate(rf"${v:.2f}$", (8.25, np.log(gV[j][keep][-1])), fontsize=6.5, va="center", color="0.25")
    ax1.set_xlim(1.7, 8.9)
    ax1.set_xlabel(r"$w$ (nm)")
    ax1.set_ylabel(r"$\ln(2t\,/\,\mathrm{eV})$")
    ax1.set_title("labelled by $V_0$ (eV)\n" r"each slope matches its own $\kappa$ to $\leq0.5\%$")

    fig.tight_layout()
    fig.savefig(f"{outdir}/log_gap.svg")


if __name__ == "__main__":
    N_LAM = 21
    h = 0.010                           # nm, grid spacing
    L = (M + 1) * h                     # psi = 0 walls sit one h outside x
    x = -L / 2 + h * np.arange(1, M + 1)
    ws = np.arange(1.0, 8.01, 0.5)
    V0s = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35])

    lamdas, Esw, two_t = sweep(x, ws, N_LAM)
    EV = np.array([sweep(x, ws, 1, V0=v)[1][:, 0, :] for v in V0s])
    run = dict(ws=ws, V0s=V0s, lamdas=lamdas, E=Esw, two_t=two_t, EV=EV)

    # Setting V0 = 0 makes the grid exactly an infinite well of width L
    n = np.arange(1, 7)
    box = {}
    for hs, Ms in ((h, M), (h / 2, 2 * M)):
        Ls = (Ms + 1) * hs
        xs = -Ls / 2 + hs * np.arange(1, Ms + 1)
        Es = track(xs, np.array([0.0]), 8.0, n.size, A, 0.0)[0][0]
        box[Ms] = (Es, HBAR2_OVER_2M * (n * np.pi / Ls) ** 2)
    Ebox, Eexact = box[M]
    rel = Ebox / Eexact - 1
    ratio = np.mean(rel / (box[2 * M][0] / box[2 * M][1] - 1))

    results = measure(x, run, A, V0)

    slope, kappa, floor = results["slope"], results["k_ind"], results["floor"]
    two_t, conv = results["two_t"], results["conv"]
    kw = abs(slope + kappa) / kappa
    kV_worst = np.max(np.abs(results["sV"] + results["kV"]) / results["kV"])
    slope_range = max(conv) - min(conv)

    print(f"grid  L = {L:.2f} nm,  h = {h} nm,  M = {M}\n")
    print(f"{'check':<16}{'quantity':<27}measured | expected")
    print(f"box energies    n = 1..6 vs analytic       {np.abs(rel).max():.1e} | O(h^2)")
    print(f"box, O(h^2)     error ratio on halving h   {ratio:.3f} | 4")
    print(f"node theorem    nodes for n = 1..8         {results['nodes']} | state n has n-1 (node theorem)")
    print(f"tracking        crossings, worst overlap   {results['swaps']} | 0," f"  {results['ovl']:.4f} | >0.9")
    print(f"two-level fit   d fit | from |psi|^2       {results['d_fit']:.4f}" f" | {results['d_psi']:.4f}")
    print(f"                2t fit | direct            " f"{results['t2_fit']:.4e} | {results['two_t_fig']:.4e}")
    se = results["slope_se"]
    print(f"LOG-GAP SLOPE   fit | kappa                {slope:+.5f} | " f"{-kappa:+.5f}   ({kw:.3%}, {abs(slope + kappa) / se:.1f} s.e.)")
    print(f"vs height       worst of {results['V0s'].size} depths" f"          {kV_worst:.2%}")
    print(f"precision       floor | smallest swept     {floor:.1e} | " f"{two_t.min():.2e} eV")
    gf = results["gap_fl"]
    print(f"                gaps at w = 11, 12, 13 nm  "
          f"{gf[0]:.1e}, {gf[1]:.1e}, {gf[2]:.1e} | last below the floor")
    print(f"convergence     slope range, h/3 and box   {slope_range:.1e} nm^-1" f"   ({slope_range / abs(np.mean(conv)):.3%})")

    figures(x, run, results)
    
