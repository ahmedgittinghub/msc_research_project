# Variational and Quantum Simulation of Donor States in ZnSe

This repo has the code for my MSc dissertation in Applied Quantum Computing at the University of Surrey (2026). I model the ground state of the outer electron of an **indium donor in zinc selenide (ZnSe)**. I compare classical diagonalisation against the **Variational Quantum Eigensolver (VQE)** and the **Non-Orthogonal Variational Quantum Eigensolver (NO-VQE)** for both accuracy and runtime.

## Why this matters

A donor atom in a semiconductor binds one weakly held electron, and that electron's spin can be used as a qubit. Donors in II-VI semiconductors like ZnO and ZnSe are interesting because they could meet three requirements at once: a spin-photon interface, a host with zero nuclear spin, and deterministic placement of the defect. Diamond NV centres and phosphorus in silicon each only meet two of these.

To predict how that spin responds to electric and magnetic fields, you first need an accurate ground state. Most of the time the electron sits far from the impurity core and behaves like a hydrogen electron. There is still a small chance of it being very close to the core, where the hydrogen picture stops working. That is what the **central-cell correction** accounts for, and it is the part of the model this project depends on most.

## Validation ladder

I built confidence in the code in three stages before running the actual target:

1. **Hydrogen**: known exact answer of -0.5 Hartree.
2. **P, As, Sb and Bi donors in silicon**: tabulated r_cc values and experimental binding energies from Saraiva et al.
3. **Indium in ZnSe**: no tabulated r_cc, so I fitted it to reproduce the measured binding energy of -28.1 meV (Merz et al.).

## Repo layout

Each basis set (STO-4G, STO-5G, STO-6G) has its own folder with the same six notebooks. The variational notebooks end in `-4g`, `-5g` or `-6g` depending on the basis.

| Notebook | What it does |
|---|---|
| `optimiser_r_cc.ipynb` | Fits r_cc so the classical ground state hits -28.1 meV |
| `classical_code.ipynb` | Builds H and S and solves the generalised eigenvalue problem with `scipy.linalg.eigh(H, S)` |
| `1-layer-vqe-<n>g.ipynb` | VQE on the Löwdin-transformed Hamiltonian, 1-layer ansatz |
| `2-layer-vqe-<n>g.ipynb` | VQE on the Löwdin-transformed Hamiltonian, 2-layer ansatz |
| `1-layer-no-vqe-<n>g.ipynb` | NO-VQE on the raw H and S, 1-layer ansatz |
| `2-layer-no-vqe-<n>g.ipynb` | NO-VQE on the raw H and S, 2-layer ansatz |

### What changes between the bases

The physics code (the integrals, building H and S) is identical in every folder. The only things that change are:

| | STO-4G | STO-5G | STO-6G |
|---|---|---|---|
| Primitives in `alpha_dict` | 4 | 5 | 6 |
| r_cc in `pe_cc_func` | 0.3542 | 0.1521 | 0.0001 |
| `nqubits` | 2 | 3 | 3 |
| Padding to 8 × 8 | No | Yes | Yes |
| Parameters (1 layer / 2 layers) | 8 / 12 | 12 / 18 | 12 / 18 |

**STO-4G** fits a 4 × 4 matrix exactly onto 2 qubits, so the matrices go straight into `qml.Hermitian` with no padding.

**STO-5G and STO-6G** need 3 qubits, so there is an extra padding step before the matrices are passed to PennyLane (see [Padding for 3 qubits](#padding-for-3-qubits)). In the VQE notebooks the padding is done *after* `S^(-1/2) H S^(-1/2)` is computed. In the NO-VQE notebooks both H and S get padded.

**`optimiser_r_cc.ipynb`** works the same way in every folder: swap in the exponents for that basis and run it. In the STO-5G version the STO-4G exponents are left in as a comment, and I used the STO-4G result (0.3542) as the starting guess. It converges to r_cc ≈ 0.15214, which is rounded to 0.1521 in the other notebooks.

**Multiple restarts:** some of the notebooks (for example `2-layer-vqe-5g.ipynb`) run the optimiser from 10 random starting points in a loop. They print the min, max and standard deviation of the energies, then take the parameters from the lowest-energy run (`np.argmin`) to build the eigenvector. This is where the min/max/std tables in the dissertation come from. With 3 qubits the VQE results can spread a lot between runs, so a single run is not always enough.

## The model

### Basis

I use STO-nG basis sets (n = 4, 5, 6), where n Gaussian primitives `e^(-α r²)` are fitted to approximate a Slater-type orbital `e^(-ζr)`. The exponents come straight from the [Basis Set Exchange](https://www.basissetexchange.org/).

Normally these primitives are contracted into one basis function with fixed coefficients. **Here each primitive is treated as its own basis function with its own coefficient**, so STO-nG gives an n × n matrix problem.

One thing to keep in mind is that no sum of Gaussians can reproduce the cusp at r = 0. The derivative of every Gaussian carries a factor of r, so the gradient at the origin is always zero. That is why the energies converge towards the exact value from above but never actually reach it.

### Effective mass theory

The donor Hamiltonian has the same form as hydrogen, just rescaled for the host lattice. The hydrogen exponents are scaled by:

```
α_host = α_H × (m* / ε_r)²
```

For ZnSe I use m* = 0.16 and ε_r = 8.8. For silicon it is m* = 0.3 and ε_r = 11.7.

### Hamiltonian

```
H = T + V_H + V_cc
```

Everything is in Hartree atomic units, with ε₀ = 1/(4π). The matrix elements between primitives i and j, with p = α_i + α_j, are:

- **Overlap:** `S_ij = (π / p)^(3/2)`
- **Kinetic:** `T_ij = (3 / m*) × α_i α_j π^(3/2) / p^(5/2)`
- **Screened Coulomb:** `V_H,ij = -1 / (2ε) × 1 / p`
- **Central cell:** `V_cc = -e²/(4πr) × (1/ε₀ - 1/ε) × e^(-r/r_cc)`

Energies are converted to meV by multiplying by 27211.38.

### Why the central-cell term uses Gauss–Laguerre quadrature

The central-cell integral does have a closed form (from Gradshteyn & Ryzhik) using a Gamma function and a parabolic cylinder function D₋ᵥ. My first attempt used that. The problem is that for small r_cc, `exp(q²/8p)` blows up while `D₋ᵥ(q/√(2p))` shrinks towards zero, and Python cannot handle that product. It returns a null value, which goes into the matrix and breaks the diagonalisation. Anything below r_cc ≈ 1.412 a.u. crashed.

So I evaluate it with 2-point Gauss–Laguerre quadrature instead:

```
V_cc,ij = -(1/ε₀ - 1/ε) × r_cc² × Σ_k w_k t_k exp(-p r_cc² t_k²)
```

with roots t = 2 ∓ √2 and weights w = 0.85355339, 0.14644661.

## Solvers

### Classical

`eigh(H, S)` solves `Hc = ESc` directly. The lowest eigenvalue is the ground state.

### Löwdin + VQE

Standard VQE only handles one matrix, but the Gaussian basis is not orthogonal, so I have a generalised problem with both H and S. The Löwdin transformation combines them:

```
H' = S^(-1/2) H S^(-1/2)
```

This keeps the eigenvalues the same, so VQE can minimise `⟨ψ(θ)|H'|ψ(θ)⟩`.

The catch is that computing S^(-1/2) means diagonalising S classically first. If the system is big enough to actually need a quantum computer, that step is not possible, which defeats the point.

### NO-VQE

NO-VQE (Sato et al.) gets around this by working with H and S directly. It uses the same circuit, but minimises the generalised Rayleigh quotient:

```
E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩ / ⟨ψ(θ)|S|ψ(θ)⟩
```

So the circuit runs twice per evaluation (once for H, once for S), but no classical diagonalisation is needed beforehand.

Both variational methods use BFGS through `scipy.optimize.minimize`, with circuits simulated in PennyLane on `default.qubit`.

## The ansatz and the parameter count

The ansatz is built from RY and RZ rotations with CNOTs between them:

- **1 layer:** RY, RZ → CNOT → RY, RZ
- **2 layers:** RY, RZ → CNOT → RY, RZ → CNOT → RY, RZ

I use RY and RZ because together they cover both angles on the Bloch sphere: RY sets θ (the amplitudes) and RZ sets φ (the phase), independently of each other. RX would move both θ and φ at once. The CNOT entangles the qubits, and without it the circuit can only make product states. Putting a CNOT between the rotation blocks also gives the circuit access to more states than the rotations alone.

### How many parameters you actually need

An N × N matrix needs ⌈log₂ N⌉ qubits. The minimum number of parameters needed to reach any state on n qubits is:

```
2^(n+1) - 2
```

This is the part that explains most of the results:

| Basis | Matrix | Qubits | Minimum needed | 1 layer | 2 layers |
|---|---|---|---|---|---|
| STO-4G | 4 × 4 | 2 | 6 | 8 ✅ | 12 ✅ |
| STO-5G | 5 × 5 | 3 | 14 | 12 ❌ | 18 ✅ |
| STO-6G | 6 × 6 | 3 | 14 | 12 ❌ | 18 ✅ |

At STO-4G, 1 layer already gives more than the 6 parameters needed, so adding a second layer does nothing for accuracy. All six methods give the same answer.

At STO-5G and STO-6G, 1 layer only gives 12 parameters against the 14 needed. Some states that are part of the solution are simply out of reach for the circuit, so it cannot hit the true ground state however well the optimiser does. With 2 layers it goes up to 18 parameters, and both VQE and NO-VQE match the classical result.

### Padding for 3 qubits

A 3-qubit register is 8-dimensional, so the 5 × 5 and 6 × 6 matrices get embedded in the top-left of an 8 × 8 matrix:

- **H** is padded with zeros off-diagonal and **1000 Hartree** on the padded diagonal. If these were zero they would sit near the real eigenvalues and mess with the optimiser, so they are set high enough to have no effect. For VQE the padding is added *after* the Löwdin transform, otherwise the physical and padded blocks get mixed.
- **S** (for NO-VQE) is padded with **1s** on the diagonal. Zeros would make S singular, and any weight in the padded block would send `⟨ψ|S|ψ⟩` to zero and the quotient to infinity.

In the code it looks like this (from the STO-5G NO-VQE notebook):

```python
pad_size = 8 - H.shape[0]
H = np.pad(H, ((0, pad_size), (0, pad_size)), mode='constant')
H[N][N] = 1000
H[N+1][N+1] = 1000
H[N+2][N+2] = 1000
S = np.pad(S, ((0, pad_size), (0, pad_size)), mode='constant')
S[N][N] = 1
S[N+1][N+1] = 1
S[N+2][N+2] = 1
```

For STO-6G there are only 2 padded rows, so only `N` and `N+1` get set.

Because the circuit works on 8 amplitudes, `qml.state()` returns an 8-component eigenvector. The padded components come out close to zero (around 1e-4 or smaller once converged). I drop them, keep the first 5 (STO-5G) or 6 (STO-6G) components, and re-normalise. The notebooks print the full 8-component vector, so this last step is done afterwards.

## Results

### Hydrogen

| Basis | Best energy (Ha) | Error (Ha) |
|---|---|---|
| STO-4G | -0.4988 | 0.0012 |
| STO-5G | -0.4996 | 0.0004 |
| STO-6G | -0.4999 | 0.0001 |

Converges from above as expected, and never hits -0.5 because of the cusp.

### Silicon donors

All solvers agree with each other (with enough parameters), but they do not match experiment. For example, at STO-4G: P -41.89, As -46.47, Sb -39.63, Bi -57.62 meV. This is a model problem and not a code problem. The silicon conduction band has six minima, while my Hamiltonian only has one valley, and the r_cc values from Saraiva et al. were fitted for the multi-valley model.

### Indium in ZnSe

Fitted r_cc values:

| Basis | r_cc (a.u.) |
|---|---|
| STO-4G | 0.3542 |
| STO-5G | 0.1521 |
| STO-6G | 0.0001 |

r_cc shrinks quickly as the basis gets bigger. At STO-6G it is so small that the central-cell term basically switches off, and the basis reaches -28.1 meV on its own. So **these r_cc values are basis-dependent** and are compensating for basis incompleteness. They are not a physical description of the indium core and should not be compared with the silicon values.

Best ground-state energy (meV) from each method over 10 restarts:

| Method | STO-4G | STO-5G | STO-6G |
|---|---|---|---|
| Classical | -28.10 | -28.10 | -28.10 |
| Classical + Löwdin | -28.10 | -28.10 | -28.10 |
| VQE, 1 layer | -28.10 | -23.71 | -23.71 |
| NO-VQE, 1 layer | -28.10 | -27.99 | -28.04 |
| VQE, 2 layers | -28.10 | -28.09 | -28.10 |
| NO-VQE, 2 layers | -28.10 | -28.10 | -28.10 |

This lines up with the parameter count. At STO-4G everything matches. At STO-5G and STO-6G, 1-layer VQE gets stuck well above the answer, 1-layer NO-VQE gets close, and both 2-layer methods hit it.

Since r_cc was fitted to hit -28.1 meV, the solvers agreeing on that number is by construction. What it does show is that the quantum solvers match the classical one when they have enough parameters.

### Runtime

- Classical diagonalisation is 3 to 4 orders of magnitude faster than any variational method at these sizes.
- **Per step**, VQE is cheaper than NO-VQE since NO-VQE runs the circuit twice.
- **Total convergence** at 3 qubits tells a different story. NO-VQE converged much faster with 2 layers: 31.09 s vs 121.4 s at STO-5G, and 53.78 s vs 79.22 s at STO-6G. VQE also had a much larger spread across restarts, which points to a landscape with more false minima that BFGS gets stuck in.
- **1-layer NO-VQE** is a good trade-off when you only need the energy. Even though it is under the parameter threshold, it got within about 0.1 meV of the classical answer.

### Eigenvectors

- VQE eigenvectors match the Löwdin coefficients, and the component with the largest exponent (closest to the core) has the smallest weight, which is what you would expect physically.
- NO-VQE and plain classical eigenvectors come out peaked in the middle. That is because the primitives are non-orthogonal and unnormalised, so the coefficients do not represent real probabilities.

**If you need the energy only, use NO-VQE. If you need meaningful coefficients, use an orthonormal basis (or VQE with Löwdin).**

## Running it

Requirements:

```
pip install numpy scipy pennylane
```

Built with Python 3.12. Open any notebook and run the cells top to bottom. The last cells print the ground-state energy in meV and the eigenvector (absolute amplitudes from `qml.state()`).

To change the basis, swap the exponents in `alpha_dict` and the r_cc value in `pe_cc_func`, and set `nqubits` to match (2 for STO-4G, 3 for STO-5G/6G, with padding). If you are adding a new basis, run `optimiser_r_cc.ipynb` first to get its r_cc, then paste that value into the other notebooks.

Note: the notebooks start from random initial angles, so a single run of a 3-qubit notebook will not always land on the same number. For example, one run of `1-layer-no-vqe-5g.ipynb` gives -27.99 meV and `1-layer-vqe-5g.ipynb` gives -23.71 meV, while the 2-layer NO-VQE run gives -28.10 meV. For the dissertation each variational method was run from ten initialisations and the lowest energy was reported (see the restart loop in `2-layer-vqe-5g.ipynb`).

## Future work

- Express the Hamiltonian in second quantisation and map it to qubits with Jordan–Wigner, so the matrix never has to be built classically.
- Orthonormalise the basis at the integral stage. That gets rid of S entirely, so there is no Löwdin bottleneck and |c_n|² becomes an actual probability.

## References

- A. L. Saraiva et al., *Theory of one and two donors in silicon*, J. Phys.: Condens. Matter 27, 154208 (2015)
- J. L. Merz et al., *Optical Properties of Substitutional Donors in ZnSe*, Phys. Rev. B 6, 545 (1972)
- Y. Sato et al., *Variational quantum algorithm for generalized eigenvalue problems and its application to the finite element method*, arXiv:2302.12602 (2023)
- P.-O. Löwdin, J. Chem. Phys. 18, 365 (1950)
- E. F. Combarro and S. González-Castillo, *A Practical Guide to Quantum Machine Learning and Quantum Optimization*, Packt (2023)
- M. Plesch and Č. Brukner, *Quantum-state preparation with universal gate decompositions*, Phys. Rev. A 83, 032302 (2011)
- I. S. Gradshteyn and I. M. Ryzhik, *Table of Integrals, Series, and Products*, 7th ed. (2007)
- M. Troyer, *Computational Quantum Physics, Exercise 7*, ETH Zürich (2012)
- [Basis Set Exchange](https://www.basissetexchange.org/)
- [PennyLane](https://pennylane.ai/)
