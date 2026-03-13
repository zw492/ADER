#pragma once
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>

#include "Euler1D.h"
#include "Grid1D.h"
#include "TimeStep.h"
#include "MusclHancock1D.h"
#include "Weno1d.H"
#include "WenoAdapter.h"
#include "ADER3_Flux.h"

namespace step23 {

inline Eigen::MatrixXd init_modified_shock_turbulence(const fv1d::Grid1D& grid,
                                                      const euler1d::Params& par,
                                                      double eps, double k)
{
    // IC from PDF Eq (2):
    // x < -4.5: [rho,u,p] = [1.515695, 0.523346, 1.805]
    // x > -4.5: [rho,u,p] = [1 + eps*sin(kx), 0, 1]
    const double x_jump = -4.5;
    Eigen::MatrixXd U0(grid.N, 3);
    for (int i = 0; i < grid.N; ++i) {
        const double x = grid.xmin + (i + 0.5) * grid.dx;
        euler1d::Vec3 W;
        if (x < x_jump)
            W << 1.515695, 0.523346, 1.805000;
        else
            W << 1.0 + eps * std::sin(k * x), 0.0, 1.0;
        U0.row(i) = euler1d::primToCons(W, par).transpose();
    }
    return U0;
}

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
    using clock_t = std::chrono::steady_clock;
    const auto wall_start = clock_t::now();
    auto last_report = wall_start;
    long long iter = 0;
    const double report_every_sec = 180.0;

    while (t < Tfinal) {
        ++iter;
        fv1d::insert_interior_into_ghosted(U, Ubc, Nghost);
        fv1d::apply_outflow_bc(Ubc, grid.N, Nghost);
        double dt = compute_dt_cfl(U, grid.dx, CFL, par);
        if (t + dt > Tfinal) dt = Tfinal - t;

        // Convert Ubc ONCE per step, reuse for all N+1 interface calls (O(N) per step)
        const VectOfVectDouble Ubc_vov = weno_adapter::eigen_to_vov(Ubc);
        std::vector<euler1d::Vec3> Fbar(grid.N + 1);
        for (int j = 0; j <= grid.N; ++j)
            Fbar[j] = ADER3_interface_flux(weno, Ubc_vov, j, Nghost, dt, par, /*alpha_diss=*/0.0);

        for (int i = 0; i < grid.N; ++i) {
            const euler1d::Vec3 Qi   = U.row(i).transpose();
            const euler1d::Vec3 Qnew = Qi - (dt / grid.dx) * (Fbar[i + 1] - Fbar[i]);
            Unp1.row(i) = Qnew.transpose();
        }
        U.swap(Unp1);
        t += dt;

        const auto now = clock_t::now();
        const double since_last = std::chrono::duration<double>(now - last_report).count();
        if (since_last >= report_every_sec) {
            const double wall_elapsed = std::chrono::duration<double>(now - wall_start).count();
            const double pct       = (Tfinal > 0.0) ? (100.0 * t / Tfinal) : 0.0;
            const double est_total = (t > 0.0) ? wall_elapsed * (Tfinal / t) : 0.0;
            const double est_left  = std::max(0.0, est_total - wall_elapsed);
            std::cout << std::defaultfloat << std::setprecision(6)
                      << "[ADER3] iter=" << iter
                      << " t=" << t << "/" << Tfinal
                      << " (" << std::setprecision(2) << pct << "%)"
                      << " dt=" << std::setprecision(6) << dt
                      << " wall=" << std::setprecision(1) << wall_elapsed << "s"
                      << " est_left=" << est_left << "s"
                      << std::endl;
            last_report = now;
        }
    }
    return U;
}

inline Eigen::MatrixXd evolve_MUSCL_outflow(const Eigen::MatrixXd& U0,
                                            const fv1d::Grid1D& grid,
                                            int Nghost,
                                            const euler1d::Params& par,
                                            double CFL,
                                            double Tfinal,
                                            muscl1d::Limiter limiter = muscl1d::Limiter::VanLeer)
{
    Eigen::MatrixXd U = U0, Ubc(grid.N + 2 * Nghost, 3), Unp1;
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

inline void dump_window_csv(const std::string& filename,
                            const Eigen::MatrixXd& U,
                            const fv1d::Grid1D& grid,
                            const euler1d::Params& par,
                            double x_min_win, double x_max_win)
{
    std::ofstream f(filename);
    f << "x,rho,u,p\n";
    for (int i = 0; i < grid.N; ++i) {
        const double x = grid.xmin + (i + 0.5) * grid.dx;
        if (x < x_min_win || x > x_max_win) continue;
        const euler1d::Vec3 W = euler1d::consToPrim(U.row(i).transpose(), par);
        f << std::setprecision(16) << x << "," << W(0) << "," << W(1) << "," << W(2) << "\n";
    }
}

inline void run()
{
    std::cout << "\n=== 3.3 Modified shock-turbulence interaction ===\n";

    euler1d::Params par;
    par.gamma = 1.4;
    par.strict_checks = true;

    const double xmin = -5.0, xmax = 5.0;
    const int    N    = 2000;
    fv1d::Grid1D grid(xmin, xmax, N);

    const double Tout   = 5.0;        // PDF Section 3.3: Tout = 5.0
    const double eps    = 0.1;
    const double k      = 20.0 * M_PI;
    const double CFL    = 0.9;
    const int    Nghost = 3;
    const double xw0 = -2.5, xw1 = 3.5;

    Eigen::MatrixXd U0 = init_modified_shock_turbulence(grid, par, eps, k);

    // ADER3 runs FIRST
    std::cout << "Running ADER3 (N=" << N << ", Tout=" << Tout << ")...\n";
    const auto t0_a = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd U_ader = evolve_ADER3_outflow(U0, grid, Nghost, par, CFL, Tout);
    const double ader_sec = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0_a).count();
    dump_window_csv("ShockTurb_ADER3.csv", U_ader, grid, par, xw0, xw1);
    std::cout << std::defaultfloat << "ADER3 finished. runtime_sec=" << ader_sec << "\n";

    // MUSCL with Van Leer limiter runs SECOND (supervisor instruction)
    std::cout << "Running MUSCL/VanLeer (N=" << N << ", Tout=" << Tout << ")...\n";
    const auto t0_m = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd U_muscl = evolve_MUSCL_outflow(U0, grid, Nghost, par, CFL, Tout,
                                                    muscl1d::Limiter::VanLeer);
    const double muscl_sec = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0_m).count();
    dump_window_csv("ShockTurb_MUSCL.csv", U_muscl, grid, par, xw0, xw1);
    std::cout << std::defaultfloat << "MUSCL/VanLeer finished. runtime_sec=" << muscl_sec << "\n";

    std::cout << "Wrote: ShockTurb_ADER3.csv and ShockTurb_MUSCL.csv\n";
    std::cout << "ADER3         runtime: " << ader_sec  << " s\n";
    std::cout << "MUSCL/VanLeer runtime: " << muscl_sec << " s\n";
}

} // namespace step23