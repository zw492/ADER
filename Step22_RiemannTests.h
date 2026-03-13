#pragma once
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

#include "Euler1D.h"
#include "Grid1D.h"
#include "TimeStep.h"
#include "MusclHancock1D.h"
#include "Weno1d.H"
#include "ADER3_Flux.h"
#include "ExactRiemannEuler1D.h"

namespace step22 {

inline void gauss_legendre_4(std::vector<double>& xi, std::vector<double>& w)
{
    xi = { -0.8611363115940526, -0.3399810435848563,
            0.3399810435848563,  0.8611363115940526 };
    w  = {  0.3478548451374539,  0.6521451548625461,
            0.6521451548625461,  0.3478548451374539 };
}

// cell-average of conserved Q from exact primitive sampler W(x,t)
template <class Sampler>
inline euler1d::Vec3 cell_average_Q(const Sampler& sampler,
                                   double xa, double xb,
                                   double t, double x0,
                                   const euler1d::Vec3& WL, const euler1d::Vec3& WR,
                                   const euler1d::Params& par)
{
    std::vector<double> xi, w;
    gauss_legendre_4(xi, w);

    euler1d::Vec3 Qbar = euler1d::Vec3::Zero();
    const double J = 0.5 * (xb - xa);
    const double xc = 0.5 * (xa + xb);

    for (int q = 0; q < (int)xi.size(); ++q) {
        const double xq = xc + J * xi[q];
        const euler1d::Vec3 Wq = sampler.sample(WL, WR, xq, t, x0);
        const euler1d::Vec3 Qq = euler1d::primToCons(Wq, par);
        Qbar += w[q] * Qq;
    }
    Qbar *= (J / (xb - xa)); // divide by cell length
    return Qbar;
}

// Initialise piecewise-constant Riemann data as cell averages
inline Eigen::MatrixXd init_riemann_cell_averages(const fv1d::Grid1D& grid,
                                                  const euler1d::Vec3& WL,
                                                  const euler1d::Vec3& WR,
                                                  double x0,
                                                  const euler1d::Params& par)
{
    exact_riemann::ExactEulerRiemann ex(par);

    Eigen::MatrixXd U0(grid.N, 3);
    for (int i = 0; i < grid.N; ++i) {
        const double xa = grid.xmin + i * grid.dx;
        const double xb = xa + grid.dx;
        const euler1d::Vec3 Qbar = cell_average_Q(ex, xa, xb, /*t=*/0.0, x0, WL, WR, par);
        U0.row(i) = Qbar.transpose();
    }
    return U0;
}

// Evolve ADER3 with OUTFLOW BC
inline Eigen::MatrixXd evolve_ADER3_outflow(const Eigen::MatrixXd& U0,
                                            const fv1d::Grid1D& grid,
                                            int Nghost,
                                            const euler1d::Params& par,
                                            double CFL,
                                            double Tfinal)
{
    WENO1d weno(/*PolyDegree=*/2, grid.dx);

    Eigen::MatrixXd U = U0;
    Eigen::MatrixXd Ubc(grid.N + 2 * Nghost, 3);
    Eigen::MatrixXd Unp1(grid.N, 3);

    double t = 0.0;
    while (t < Tfinal) {
        fv1d::insert_interior_into_ghosted(U, Ubc, Nghost);
        fv1d::apply_outflow_bc(Ubc, grid.N, Nghost);

        double dt = compute_dt_cfl(U, grid.dx, CFL, par);
        if (t + dt > Tfinal) dt = Tfinal - t;

        std::vector<euler1d::Vec3> Fbar(grid.N + 1);
        for (int j = 0; j <= grid.N; ++j) {
            Fbar[j] = ADER3_interface_flux(weno, Ubc, j, Nghost, dt, par, /*alpha_diss=*/0.0);
        }

        for (int i = 0; i < grid.N; ++i) {
            const euler1d::Vec3 Qi = U.row(i).transpose();
            const euler1d::Vec3 Qnew = Qi - (dt / grid.dx) * (Fbar[i + 1] - Fbar[i]);
            Unp1.row(i) = Qnew.transpose();
        }

        U.swap(Unp1);
        t += dt;
    }
    return U;
}

// Evolve MUSCL–Hancock with OUTFLOW BC (reuses muscl1d::step with bcType)
inline Eigen::MatrixXd evolve_MUSCL_outflow(const Eigen::MatrixXd& U0,
                                            const fv1d::Grid1D& grid,
                                            int Nghost,
                                            const euler1d::Params& par,
                                            double CFL,
                                            double Tfinal,
                                            muscl1d::Limiter limiter = muscl1d::Limiter::MC)
{
    Eigen::MatrixXd U = U0;
    Eigen::MatrixXd Ubc(grid.N + 2 * Nghost, 3);
    Eigen::MatrixXd Unp1;

    double t = 0.0;
    while (t < Tfinal) {
        fv1d::insert_interior_into_ghosted(U, Ubc, Nghost);
        fv1d::apply_outflow_bc(Ubc, grid.N, Nghost);

        double dt = compute_dt_cfl(U, grid.dx, CFL, par);
        if (t + dt > Tfinal) dt = Tfinal - t;

        muscl1d::step(Ubc, grid.N, Nghost, dt, grid.dx, par, Unp1,
              limiter, muscl1d::BcType::Outflow,
              muscl1d::FluxType::GodunovExact);

        U.swap(Unp1);
        t += dt;
    }
    return U;
}

// Compute density errors between numerical U (cell averages) and exact cell averages at time T
inline void density_errors(const Eigen::MatrixXd& U_num,
                           const fv1d::Grid1D& grid,
                           const euler1d::Vec3& WL, const euler1d::Vec3& WR,
                           double x0, double T,
                           const euler1d::Params& par,
                           double& L1, double& L2, double& Linf)
{
    exact_riemann::ExactEulerRiemann ex(par);

    double s1 = 0.0, s2 = 0.0, sInf = 0.0;

    for (int i = 0; i < grid.N; ++i) {
        const double xa = grid.xmin + i * grid.dx;
        const double xb = xa + grid.dx;

        const euler1d::Vec3 Qex = cell_average_Q(ex, xa, xb, T, x0, WL, WR, par);
        const double rho_ex = Qex(0);
        const double rho_n  = U_num(i, 0);
        const double diff = rho_n - rho_ex;

        s1   += std::abs(diff) * grid.dx;
        s2   += diff * diff * grid.dx;
        sInf  = std::max(sInf, std::abs(diff));
    }

    L1 = s1;
    L2 = std::sqrt(s2);
    Linf = sInf;
}

// Dump CSV with numerical + exact primitive at cell centers (for plotting)
inline void dump_csv(const std::string& filename,
                     const Eigen::MatrixXd& U_num,
                     const fv1d::Grid1D& grid,
                     const euler1d::Vec3& WL, const euler1d::Vec3& WR,
                     double x0, double T,
                     const euler1d::Params& par)
{
    exact_riemann::ExactEulerRiemann ex(par);

    std::ofstream f(filename);
    f << "x,rho_num,u_num,p_num,rho_ex,u_ex,p_ex\n";
    for (int i = 0; i < grid.N; ++i) {
        const double x = grid.xmin + (i + 0.5) * grid.dx;

        const euler1d::Vec3 Qi = U_num.row(i).transpose();
        const euler1d::Vec3 Wn = euler1d::consToPrim(Qi, par);
        const euler1d::Vec3 We = ex.sample(WL, WR, x, T, x0);

        f << std::setprecision(16)
          << x << ","
          << Wn(0) << "," << Wn(1) << "," << Wn(2) << ","
          << We(0) << "," << We(1) << "," << We(2) << "\n";
    }
}

// Run one named test
inline void run_test(const std::string& name,
                     const euler1d::Vec3& WL,
                     const euler1d::Vec3& WR,
                     double x0, double Tout,
                     int N,
                     double CFL,
                     const euler1d::Params& par,
                     bool also_run_muscl = true)
{
    // Domain choice
    fv1d::Grid1D grid(0.0, 1.0, N);
    const int Nghost = 3;

    std::cout << "\n=== 2.2 Riemann test: " << name << " ===\n";
    std::cout << "N=" << N << ", x0=" << x0 << ", Tout=" << Tout << "\n";

    Eigen::MatrixXd U0 = init_riemann_cell_averages(grid, WL, WR, x0, par);

    // ADER3
    Eigen::MatrixXd U_ader = evolve_ADER3_outflow(U0, grid, Nghost, par, CFL, Tout);
    double L1, L2, Linf;
    density_errors(U_ader, grid, WL, WR, x0, Tout, par, L1, L2, Linf);
    std::cout << "ADER3 density errors: L1=" << L1 << "  L2=" << L2 << "  Linf=" << Linf << "\n";
    dump_csv(name + "_ADER3.csv", U_ader, grid, WL, WR, x0, Tout, par);

    if (also_run_muscl) {
        Eigen::MatrixXd U_muscl = evolve_MUSCL_outflow(U0, grid, Nghost, par, CFL, Tout, muscl1d::Limiter::MC);
        density_errors(U_muscl, grid, WL, WR, x0, Tout, par, L1, L2, Linf);
        std::cout << "MUSCL density errors: L1=" << L1 << "  L2=" << L2 << "  Linf=" << Linf << "\n";
        dump_csv(name + "_MUSCL.csv", U_muscl, grid, WL, WR, x0, Tout, par);
    }

    std::cout << "Wrote CSVs: " << name << "_ADER3.csv"
              << (also_run_muscl ? (" and " + name + "_MUSCL.csv") : "")
              << "\n";
}

} // namespace step22