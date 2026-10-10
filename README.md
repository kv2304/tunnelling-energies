# Quantised energies and tunnelling decay

Finite-difference solver for the time-independent Schrödinger equation in a double well with
varying potential. Computes eigenvalues and eigenstates and recovers the wavefunction's
tunnelling decay constant, $\kappa$, from the collapse of the level splitting with barrier width,
agreeing with the analytical solution $\kappa = \sqrt{2m_e(V_0-E)}/\hbar$ to 0.001 %.

## Physical concepts

A bound electron occupies the eigenstates of the Hamiltonian $\hat{H}$. Its allowed energies are
the eigenvalues of that operator, and normalisability makes them discrete.

```math
\hat{H}\psi = E\psi, \qquad
\hat{H} = -\frac{\hbar^{2}}{2m_e}\frac{\mathrm{d}^{2}}{\mathrm{d}x^{2}} + V(x)
```

Where $E < V(x)$ the eigenfunction decays instead of oscillating, with real exponent $\kappa$
set by how far the state lies below the barrier top $V_0$.

```math
\psi \sim e^{-\kappa x}, \qquad \kappa = \frac{\sqrt{2m_e(V_0-E)}}{\hbar}
```

A symmetric double well gives $[\hat{H}, \hat{\Pi}] = 0$ with $\hat{\Pi}$ the parity operator,
so the eigenstates are the even and odd superpositions of the orbitals $\phi_L$ and $\phi_R$
localised in each well. The tunnelling energy $t$ is the matrix element of $\hat{H}$ between
those two orbitals, set by their overlap inside the barrier.

```math
t = -\int \phi_L^{*}\,\hat{H}\,\phi_R\,\mathrm{d}x = -\langle \phi_L | \hat{H} | \phi_R \rangle
```

The decaying tails make $t$ non-zero, so the two wells are not degenerate: the even and odd
states are separated by $E_A - E_S = 2t$. Tilting the wells breaks parity and carries the pair
through an avoided crossing of minimum separation $2t$.

## Results

Complete method and analysis can be read in [docs/](docs/).

| | measured | independent | agreement |
|---|---|---|---|
| $\kappa$ (nm⁻¹) | 2.17469 from the fitted slope | 2.17471 from $V_0-E$ | **0.001 %** |
| $2t$ (eV) | 1.7815e-03 | 1.7813e-03 at $\lambda=0$ | 0.01 % |
| $d$ | 0.8087 | 0.8088 from the wavefunction | 0.01 % |
| $\kappa$ at six well depths | six slopes | six values from $V_0-E$ | worst 0.47 % |
| $\kappa$ stability, three grids (nm⁻¹) | range 3.4e-05 | $h/3$ and a wider box | 0.002 % of $\kappa$ |
| box energies, $n=1\ldots6$ | worst 6.7e-06 | $E_n = n^2\pi^2\hbar^2/2m_eL^2$ | $\leq$ 7e-06 |
| $O(h^2)$ truncation | error ratio 4.0031 on halving $h$ | 4 for a three-point stencil | 0.08 % |
| nodes of the lowest 8 states | 0,1,2,3,4,5,6,7 | node theorem gives $n-1$ for state $n$ | exact |
| smallest gap swept | 3.84e-09 eV | float64 floor 3.4e-13 eV | 11000x above the floor |

## Run

```
git clone https://github.com/kv2304/tunnelling-energies.git
cd tunnelling-energies
pip install -r requirements.txt
python levels.py     # ~3s, redraws figures/
```
