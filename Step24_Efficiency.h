#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <chrono>

#include "Euler1D.h"
#include "Grid1D.h"
#include "TimeStep.h"
#include "MusclHancock1D.h"
#include "ADER3_Flux.h"
#include "WenoAdapter.h"
#include "Weno1d.H"

// Task 3.4 — Efficiency analysis
//
// PDF problem (1): rho = 2 + sin^4(pi*x), u = p = 1, periodic on [-1,1]
// Tout = 100  (supervisor correction: NOT 1000)
//
// Efficiency plot:
//   x-axis: CPU time (log scale)
//   y-axis: L2 error (log scale)
//   horizontal line: L2 = 1e-3
//   Each scheme: fit log-log least-squares line through data points,
//   read off crossing time with threshold line.
//
// ADER3 sweep: N = 50, 75, 100, 125, 150, 200, 250, 300
// MUSCL sweep (Limiter::None): N = 100, 200, 400, 600, 800, 1000
//
// N=100 MUSCL/ADER3 also run for Part A visual comparison.

namespace step24 {

struct Timer {
    std::chrono::high_resolution_clock::time_point t0;
    void start() { t0 = std::chrono::high_resolution_clock::now(); }
    double stop_sec() const {
        return std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count();
    }
};

// Exact smooth solution: rho(x,t) = 2 + sin^4(pi*(x-t)), u=p=1, periodic on [-1,1]
inline euler1d::Vec3 exact_prim_smooth(double x, double t)
{
    double xx = x - t;
    const double L = 2.0;
    xx = std::fmod(xx + 1.0, L);
    if (xx < 0.0) xx += L;
    xx -= 1.0;
    const double s   = std::sin(M_PI * xx);
    const double rho = 2.0 + std::pow(s, 4.0);
    euler1d::Vec3 W; W << rho, 1.0, 1.0;
    return W;
}

// 4-point Gauss-Legendre quadrature for cell average of exact rho
inline void gauss4(std::vector<double>& xi, std::vector<double>& w)
{
    xi = { -0.8611363115940526, -0.3399810435848563,
            0.3399810435848563,  0.8611363115940526 };
    w  = {  0.3478548451374539,  0.6521451548625461,
            0.6521451548625461,  0.3478548451374539 };
}

inline double cell_avg_rho_exact(double xa, double xb, double t)
{
    std::vector<double> xi, w; gauss4(xi, w);
    const double J = 0.5*(xb-xa), xc = 0.5*(xa+xb);
    double acc = 0.0;
    for (int q = 0; q < 4; ++q)
        acc += w[q] * exact_prim_smooth(xc + J*xi[q], t)(0);
    return J * acc / (xb - xa);
}

inline Eigen::MatrixXd init_smooth_cell_averages(const fv1d::Grid1D& grid,
                                                 const euler1d::Params& par)
{
    Eigen::MatrixXd U0(grid.N, 3);
    for (int i = 0; i < grid.N; ++i) {
        const double xa = grid.xmin + i*grid.dx, xb = xa + grid.dx;
        euler1d::Vec3 W; W << cell_avg_rho_exact(xa, xb, 0.0), 1.0, 1.0;
        U0.row(i) = euler1d::primToCons(W, par).transpose();
    }
    return U0;
}

inline double L2_rho_error(const Eigen::MatrixXd& U,
                           const fv1d::Grid1D& grid, double t)
{
    double s2 = 0.0;
    for (int i = 0; i < grid.N; ++i) {
        const double xa = grid.xmin + i*grid.dx, xb = xa + grid.dx;
        const double diff = U(i,0) - cell_avg_rho_exact(xa, xb, t);
        s2 += diff*diff*grid.dx;
    }
    return std::sqrt(s2);
}

// Visual comparison CSV for Part A: x, rho_ADER3, rho_MUSCL, rho_exact
inline void dump_visual_csv(const std::string& filename,
                            const Eigen::MatrixXd& U_ader,
                            const Eigen::MatrixXd& U_muscl,
                            const fv1d::Grid1D& grid, double t)
{
    std::ofstream f(filename);
    f << "x,rho_ADER3,rho_MUSCL,rho_exact\n";
    for (int i = 0; i < grid.N; ++i) {
        const double xc = grid.xmin + (i+0.5)*grid.dx;
        f << std::setprecision(16)
          << xc << ","
          << U_ader(i,0) << ","
          << U_muscl(i,0) << ","
          << exact_prim_smooth(xc, t)(0) << "\n";
    }
}

// MUSCL with Limiter::None (good for smooth problems)
inline Eigen::MatrixXd evolve_MUSCL_periodic(const Eigen::MatrixXd& U0,
                                             const fv1d::Grid1D& grid,
                                             int Nghost,
                                             const euler1d::Params& par,
                                             double CFL, double Tfinal)
{
    Eigen::MatrixXd U = U0, Ubc(grid.N + 2*Nghost, 3), Unp1;
    double t = 0.0;
    while (t < Tfinal) {
        fv1d::insert_interior_into_ghosted(U, Ubc, Nghost);
        fv1d::apply_periodic_bc(Ubc, grid.N, Nghost);
        double dt = compute_dt_cfl(U, grid.dx, CFL, par);
        if (t + dt > Tfinal) dt = Tfinal - t;
        // Limiter::None on smooth problems
        muscl1d::step(Ubc, grid.N, Nghost, dt, grid.dx, par, Unp1,
                      muscl1d::Limiter::None, muscl1d::BcType::Periodic,
                      muscl1d::FluxType::GodunovExact);
        U.swap(Unp1);
        t += dt;
    }
    return U;
}

// ADER3 with alpha_diss=0
inline Eigen::MatrixXd evolve_ADER3_periodic(const Eigen::MatrixXd& U0,
                                             const fv1d::Grid1D& grid,
                                             int Nghost,
                                             const euler1d::Params& par,
                                             double CFL, double Tfinal)
{
    WENO1d weno(/*PolyDegree=*/2, grid.dx);
    Eigen::MatrixXd U = U0, Ubc(grid.N + 2*Nghost, 3), Unp1(grid.N, 3);
    double t = 0.0;
    int stepCount = 0;
    const auto tStart = std::chrono::high_resolution_clock::now();

    while (t < Tfinal) {
        fv1d::insert_interior_into_ghosted(U, Ubc, Nghost);
        fv1d::apply_periodic_bc(Ubc, grid.N, Nghost);
        double dt = compute_dt_cfl(U, grid.dx, CFL, par);
        if (t + dt > Tfinal) dt = Tfinal - t;

        const VectOfVectDouble Ubc_vov = weno_adapter::eigen_to_vov(Ubc);
        std::vector<euler1d::Vec3> Fbar(grid.N + 1);
        for (int j = 0; j <= grid.N; ++j)
            // alpha_diss=0: pure ADER
            Fbar[j] = ADER3_interface_flux(weno, Ubc_vov, j, Nghost, dt, par, /*alpha_diss=*/0.0);

        for (int i = 0; i < grid.N; ++i) {
            const euler1d::Vec3 Qi = U.row(i).transpose();
            Unp1.row(i) = (Qi - (dt/grid.dx)*(Fbar[i+1] - Fbar[i])).transpose();
        }
        U.swap(Unp1);
        t += dt;
        ++stepCount;

        if (stepCount % 2000 == 0) {
            const double elapsed = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - tStart).count();
            const double frac   = t / Tfinal;
            const double eta_min = (frac > 0.0) ? (elapsed/frac - elapsed)/60.0 : 0.0;
            std::cout << std::defaultfloat << std::setprecision(6)
                      << "[ADER3] t=" << t
                      << "  step=" << stepCount
                      << "  dt=" << dt
                      << "  elapsed=" << std::setprecision(3) << elapsed/60.0 << " min"
                      << "  ETA~" << std::setprecision(2) << eta_min << " min"
                      << std::endl;
        }
    }
    return U;
}

// Log-log least-squares fit: log10(time) = a + b * log10(error)
inline bool loglog_fit(const std::vector<double>& err,
                       const std::vector<double>& time,
                       double& slope_b, double& intercept_a)
{
    std::vector<double> X, Y;
    for (size_t i = 0; i < err.size(); ++i)
        if (err[i] > 0 && time[i] > 0) {
            X.push_back(std::log10(err[i]));
            Y.push_back(std::log10(time[i]));
        }
    if (X.size() < 2) return false;
    double sx=0,sy=0,sxx=0,sxy=0;
    for (size_t i=0;i<X.size();++i){sx+=X[i];sy+=Y[i];sxx+=X[i]*X[i];sxy+=X[i]*Y[i];}
    const double n=X.size(), denom=n*sxx-sx*sx;
    if (std::abs(denom)<1e-16) return false;
    slope_b=(n*sxy-sx*sy)/denom;
    intercept_a=(sy-slope_b*sx)/n;
    return true;
}

inline void run()
{
    std::cout << "\n=== 3.4 Efficiency analysis (smooth test, Tout=100) ===\n";

    euler1d::Params par;
    par.gamma = 1.4; par.strict_checks = true;

    const double xmin=-1.0, xmax=1.0;
    const double Tout   = 100.0;   // Supervisor correction: Tout=100, NOT 1000
    const double CFL    = 0.9;
    const int    Nghost = 3;
    const double targetL2 = 1e-3;

    // Open CSV early: flush after each run so data is safe even if killed
    std::ofstream fcsv("Efficiency_T100.csv", std::ios::app);  // append, not overwrite

    // Part A: Visual comparison at N=100, Tout=100
    // PDF: "results for a mesh of 100 cells should be visually compared with
    //       those obtained using the MUSCL-Hancock scheme and the exact solution"
    // Expected: ADER3 preserves oscillations clearly; MUSCL shows heavy diffusion

    std::cout << "\n--- Part A: Visual comparison at N=100, Tout=" << Tout << " ---\n";
    {
        const int N_vis = 100;
        fv1d::Grid1D grid(xmin, xmax, N_vis);
        Eigen::MatrixXd U0 = init_smooth_cell_averages(grid, par);

        std::cout << "  ADER3 N=100 running (~4 min)...\n" << std::flush;
        Timer T; T.start();
        Eigen::MatrixXd U_ader = evolve_ADER3_periodic(U0, grid, Nghost, par, CFL, Tout);
        const double t_ader = T.stop_sec();
        const double L2_ader = L2_rho_error(U_ader, grid, Tout);
        std::cout << std::defaultfloat << std::setprecision(6)
                  << "  ADER3  N=100: L2=" << L2_ader << "  time=" << t_ader << " s\n";
        fcsv << "ADER3," << N_vis << "," << std::setprecision(16) << grid.dx
             << "," << L2_ader << "," << t_ader << "\n"; fcsv.flush();

        std::cout << "  MUSCL N=100 running (~1 min)...\n" << std::flush;
        T.start();
        Eigen::MatrixXd U_muscl = evolve_MUSCL_periodic(U0, grid, Nghost, par, CFL, Tout);
        const double t_muscl = T.stop_sec();
        const double L2_muscl = L2_rho_error(U_muscl, grid, Tout);
        std::cout << std::defaultfloat << std::setprecision(6)
                  << "  MUSCL  N=100: L2=" << L2_muscl << "  time=" << t_muscl << " s\n";
        fcsv << "MUSCL," << N_vis << "," << std::setprecision(16) << grid.dx
             << "," << L2_muscl << "," << t_muscl << "\n"; fcsv.flush();

        dump_visual_csv("Visual_N100_T100.csv", U_ader, U_muscl, grid, Tout);
        std::cout << "  Wrote: Visual_N100_T100.csv\n";
    }
    
    // Part B: Efficiency sweep
    // ADER3: N = 50, 75, 100, 125, 150, 200, 250, 300
    // MUSCL: N = 100, 200, 400, 600, 800, 1000
    // Shallower ADER3 slope = reaches target faster = more efficient

    std::cout << "\n--- Part B: Efficiency sweep ---\n";
    std::cout << "    Theory: ADER3 slope~-0.667, MUSCL slope~-1.0 in log10(time) vs log10(err)\n\n";

    const std::vector<int> Ns_ader_sweep  = {50, 75, 100, 125, 150, 200, 250, 300};
    const std::vector<int> Ns_muscl_sweep = {100, 200, 400, 600, 800, 1000};

    // Storage for fit
    std::vector<double> err_a, time_a;
    std::vector<int>    Ns_a_done;
    std::vector<double> err_m, time_m;
    std::vector<int>    Ns_m_done;

    // ADER3 sweep
    // Run smaller N first
    for (int N : Ns_ader_sweep) {
        fv1d::Grid1D grid(xmin, xmax, N);
        Eigen::MatrixXd U0 = init_smooth_cell_averages(grid, par);
        std::cout << "  ADER3 N=" << N << " running...\n" << std::flush;
        Timer T; T.start();
        Eigen::MatrixXd U = evolve_ADER3_periodic(U0, grid, Nghost, par, CFL, Tout);
        const double runtime = T.stop_sec();
        const double L2 = L2_rho_error(U, grid, Tout);
        fcsv << "ADER3," << N << "," << std::setprecision(16) << grid.dx
             << "," << L2 << "," << runtime << "\n"; fcsv.flush();
        std::cout << std::defaultfloat << std::setprecision(6)
                  << "  ADER3 N=" << N
                  << "  L2=" << L2
                  << "  time=" << runtime << " s"
                  << (L2 <= targetL2 ? "  <-- BELOW TARGET" : "")
                  << "\n";
        err_a.push_back(L2); time_a.push_back(runtime); Ns_a_done.push_back(N);
    }

    // MUSCL sweep
    for (int N : Ns_muscl_sweep) {
        fv1d::Grid1D grid(xmin, xmax, N);
        Eigen::MatrixXd U0 = init_smooth_cell_averages(grid, par);
        std::cout << "  MUSCL N=" << N << " running...\n" << std::flush;
        Timer T; T.start();
        Eigen::MatrixXd U = evolve_MUSCL_periodic(U0, grid, Nghost, par, CFL, Tout);
        const double runtime = T.stop_sec();
        const double L2 = L2_rho_error(U, grid, Tout);
        fcsv << "MUSCL," << N << "," << std::setprecision(16) << grid.dx
             << "," << L2 << "," << runtime << "\n"; fcsv.flush();
        std::cout << std::defaultfloat << std::setprecision(6)
                  << "  MUSCL N=" << N
                  << "  L2=" << L2
                  << "  time=" << runtime << " s"
                  << (L2 <= targetL2 ? "  <-- BELOW TARGET" : "")
                  << "\n";
        err_m.push_back(L2); time_m.push_back(runtime); Ns_m_done.push_back(N);
    }

    fcsv.close();
    std::cout << "\nWrote: Efficiency_T100.csv\n\n";


    // Summary: report crossings and fit slopes
    auto report = [&](const std::vector<int>& Ns_,
                      const std::vector<double>& err_,
                      const std::vector<double>& time_,
                      const std::string& name) {
        std::cout << name << " results:\n";
        for (size_t i = 0; i < Ns_.size(); ++i)
            std::cout << "  N=" << Ns_[i]
                      << "  L2=" << std::setprecision(6) << err_[i]
                      << "  time=" << time_[i] << " s\n";

        // Check if any point actually crossed
        bool hit = false;
        for (size_t i = 0; i < err_.size(); ++i)
            if (err_[i] <= targetL2) { hit = true; break; }
        if (!hit) std::cout << "  (no point reached L2 <= " << targetL2 << " -- use extrapolation)\n";

        double b, a;
        if (loglog_fit(err_, time_, b, a)) {
            const double t_ex = std::pow(10.0, a + b * std::log10(targetL2));
            std::cout << std::defaultfloat << std::setprecision(4)
                      << "  log-log slope = " << b
                      << "  (theory: ADER3=-0.667, MUSCL=-1.0)\n"
                      << "  extrapolated CPU time at L2=1e-3: "
                      << t_ex << " s  (" << t_ex/3600.0 << " hr)\n";
        }
        std::cout << "\n";
    };

    // Part A stored in CSV; slopes/fit printed below using sweep-only data
    // Note: for the actual plot, read all rows from Efficiency_T100.csv
    report(Ns_a_done, err_a, time_a, "ADER3 (sweep only; add N=100 from Part A for full fit)");
    report(Ns_m_done, err_m, time_m, "MUSCL (sweep only; N=100 Part A also in CSV)");

    std::cout << "NOTE: For the final efficiency plot, load ALL rows from\n";
    std::cout << "      Efficiency_T100.csv (includes Part A N=100 for both schemes).\n";
    std::cout << "      Use ADER3 rows with L2 in asymptotic range for the fit.\n";
    std::cout << "      MUSCL N=100 may be pre-asymptotic at T=100 -- check visually.\n";
}

} // namespace step24