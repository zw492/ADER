# ADER
The ADER approach for the one-dimensional Euler equations. Advisor: Riccardo Dematté

## Dependencies

- **Eigen 3** (header-only linear algebra library)  
  Download from https://eigen.tuxfamily.org and note the path to the root directory.
- C++17-capable compiler (GCC ≥ 9 or Clang ≥ 10)

---

## Building

```bash
g++ -O0 -std=c++17 -I/path/to/eigen \
    mainADER.cpp Weno1d.cpp \
    -o mainADER
```

Replace `/path/to/eigen` with the actual path to the Eigen root directory (the folder containing the `Eigen/` subdirectory).  
All other source files are header-only and are `#include`d directly by `mainADER.cpp`.

---

## Running

```bash
./mainADER
```

The executable runs all verification steps and all four tasks sequentially in a single invocation.

---

## Output CSV files

All CSV files are written to the working directory.

| CSV file | Written by | Contents |
|---|---|---|
| `convergence_table_T2.csv` | Step 8 / Task 3.1 | `scheme, N, dx, L1, L2, Linf, rate_L1, rate_L2, rate_Linf, runtime_sec` — density error norms and convergence rates at T=2 for N=25,50,100,200,400 |
| `Sod_ADER3.csv`, `Sod_MUSCL.csv` | Task 3.2 | `x, rho, u, p, rho_exact, u_exact, p_exact` — cell-centre primitives + exact solution at T=0.2, N=200 |
| `Lax_ADER3.csv`, `Lax_MUSCL.csv` | Task 3.2 | Same format, T=0.14, N=200 |
| `ShockTurb_ADER3.csv`, `ShockTurb_MUSCL.csv` | Task 3.3 | `x, rho, u, p` — windowed output over x∈[−2.5, 3.5] at T=5, N=2000 |
| `Efficiency_T100.csv` | Task 3.4 | `scheme, N, dx, L2, runtime_sec` — L² density error and CPU time at T=100; opened in **append mode** (safe to resume if killed) |
| `Visual_N100_T100.csv` | Task 3.4 Part A | `x, rho_ADER3, rho_MUSCL, rho_exact` — density profiles at T=100, N=100 |

---

## Generating figures

Once all CSV files have been produced, run:

```bash
python plot_all_ADER.py
```

Requires Python 3 with NumPy, pandas, Matplotlib, and SciPy. The script reads each CSV file listed above and writes all report figures to the `figures/` directory.

---

## Default numerical settings

### Shared

| Parameter | Value | Where set |
|---|---|---|
| CFL number ν | **0.9** | All tasks, both schemes |
| Ghost cells N_g | **3** | All tasks |
| Equation of state γ | **1.4** | `Euler1D.h` |
| Interface flux (MUSCL) | **Exact Godunov** (`FluxType::GodunovExact`) | All tasks |
| ADER additional dissipation | **α_diss = 0.0** | All tasks |

### ADER3 flux (`ADER3_Flux.h`)

- **WENO reconstruction**: degree M=2 (`WENO1d(PolyDegree=2, dx)`), three candidate stencils, fifth-order spatial accuracy in smooth regions. Supplies interface values Q^{L,R} and first/second spatial derivatives Q^{L,R}_x, Q^{L,R}_{xx} at each interface.
- **Leading-order Riemann solve**: Toro exact Euler solver (`ExactRiemannEuler1D.h`) gives Godunov state Q₀ = R(0; Q^L, Q^R).
- **Characteristic upwinding**: eigensystem of A(Q₀) — the Jacobian evaluated at Q₀, not a Roe average — upwinds the reconstructed spatial derivatives.
- **CK procedure** (`CK_Euler.h`): computes Q_t and Q_tt from (Q₀, Q⁰_x, Q⁰_xx) via the PDE identity Q_t = −A Q_x.
- **Time-averaged flux**: **2-point Gauss–Legendre quadrature** on [0, Δt] with nodes γ_{1,2} = (Δt/2)(1 ∓ 1/√3) and equal weights ω = 1/2. Integrates the quadratic-in-τ integrand F(J(τ)) to third-order accuracy. The nonlinear flux F is evaluated directly at each quadrature state without linearisation.

### MUSCL–Hancock limiter choices by task

| Task | MUSCL limiter |
|---|---|
| 3.1 Convergence | None (unlimited linear reconstruction) |
| 3.2 Riemann tests (Sod, Lax) | MC (Monotonized Central) |
| 3.3 Shock–turbulence | Van Leer |
| 3.4 Efficiency | None |

---

## Task configurations at a glance

| Task | Domain | N | T_out | BC | IC |
|---|---|---|---|---|---|
| 3.1 Convergence | [−1, 1] | 25,50,100,200,400 | 2.0 | Periodic | ρ = 2+sin⁴(πx), u=1, p=1 |
| 3.2 Sod | [0, 1] | 200 | 0.2 | Outflow | ρ_L=1, u_L=0, p_L=1; ρ_R=0.125, u_R=0, p_R=0.1; x₀=0.5 |
| 3.2 Lax | [0, 1] | 200 | 0.14 | Outflow | ρ_L=0.445, u_L=0.698, p_L=3.528; ρ_R=0.5, u_R=0, p_R=0.571; x₀=0.6 |
| 3.3 Shock–turbulence | [−5, 5] | 2000 | 5.0 | Outflow | Left shock state / right ρ=1+0.1 sin(20πx), u=0, p=1 |
| 3.4 Efficiency | [−1, 1] | ADER: 50–300; MUSCL: 100–1000 | 100.0 | Periodic | Same as 3.1 |

---

## Source file overview

| File | Role |
|---|---|
| `mainADER.cpp` | Entry point; runs Steps 1–8 then Tasks 3.2–3.4 sequentially |
| `Euler1D.h` | Primitive↔conserved conversions, flux F(Q), sound speed, Params struct |
| `Grid1D.h` | Uniform grid, ghost-cell allocation, periodic/outflow BC fills |
| `TimeStep.h` | CFL time-step controller: Δt = ν Δx / a_max |
| `ExactRiemannEuler1D.h` | Toro exact Riemann solver (Newton iteration on pressure function) |
| `RiemannFlux.h` | Rusanov flux and exact Godunov flux dispatch |
| `Weno1d.H` / `Weno1d.cpp` | Fifth-order WENO reconstruction (PolyDegree=2) |
| `WenoAdapter.h` | Eigen↔WENO internal vector conversion |
| `CK_Euler.h` | Cauchy–Kowalewski procedure: Q_t and Q_tt from (Q₀, Q⁰_x, Q⁰_xx) |
| `ADER3_Flux.h` | Complete TT-GRP ADER3 interface flux assembly |
| `MusclHancock1D.h` | MUSCL–Hancock scheme with None/Minmod/MC/Van Leer limiters |
| `Step8_Verification.h` | Task 3.1 convergence study; writes `convergence_table_T2.csv` |
| `Step22_RiemannTests.h` | Task 3.2 Sod and Lax tests; writes `{Sod,Lax}_{ADER3,MUSCL}.csv` |
| `Step23_ShockTurbulence.h` | Task 3.3 shock–turbulence; writes `ShockTurb_{ADER3,MUSCL}.csv` |
| `Step24_Efficiency.h` | Task 3.4 efficiency sweep; writes `Efficiency_T100.csv` and `Visual_N100_T100.csv` |
