#pragma once
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <chrono>
#include <iostream>
#include <iomanip>

#include "Euler1D.h"
#include "Grid1D.h"
#include "TimeStep.h"
#include "MusclHancock1D.h"
#include "Weno1d.H"
#include "ADER3_Flux.h"

namespace step8 {

// Gauss–Legendre 4-pt on [-1,1]
inline void gauss_legendre_4(std::vector<double>& xi, std::vector<double>& w)
{
    xi = { -0.8611363115940526, -0.3399810435848563,
            0.3399810435848563,  0.8611363115940526 };
    w  = {  0.3478548451374539,  0.6521451548625461,
            0.6521451548625461,  0.3478548451374539 };
}

// wrap x into [xmin, xmax)
inline double wrap_periodic(double x, double xmin, double xmax)
{
    const double L = xmax - xmin;
    double y = std::fmod(x - xmin, L);
    if (y < 0.0) y += L;
    return xmin + y;
}

// smooth periodic test
// rho = 1 + a sin(2pi (x-u0 t)), u=u0, p=p0

inline euler1d::Vec3 smooth_primitive(double x, double t,
                                      double xmin, double xmax,
                                      double a, double u0, double p0)
{
    const double PI = 3.14159265358979323846;
    double xshift = wrap_periodic(x - u0 * t, xmin, xmax);
    double rho = 1.0 + a * std::sin(2.0 * PI * (xshift - xmin) / (xmax - xmin));
    euler1d::Vec3 W; W << rho, u0, p0;
    return W;
}

inline euler1d::Vec3 cell_average_Q(const fv1d::Grid1D& grid, int i, double t,
                                    double a, double u0, double p0,
                                    const euler1d::Params& par)
{
    std::vector<double> xi, w;
    gauss_legendre_4(xi, w);

    const double xc = grid.xc[i];
    const double dx = grid.dx;

    euler1d::Vec3 Qavg = euler1d::Vec3::Zero();

    for (size_t q = 0; q < xi.size(); ++q) {
        const double xq = xc + 0.5 * dx * xi[q];
        euler1d::Vec3 Wq = smooth_primitive(xq, t, grid.xmin, grid.xmax, a, u0, p0);
        euler1d::Vec3 Qq = euler1d::primToCons(Wq, par);
        Qavg += w[q] * Qq;
    }
    Qavg *= 0.5;
    return Qavg;
}

inline Eigen::MatrixXd init_smooth_cell_averages(const fv1d::Grid1D& grid,
                                                 double a, double u0, double p0,
                                                 const euler1d::Params& par)
{
    Eigen::MatrixXd U(grid.N, 3);
    for (int i = 0; i < grid.N; ++i) {
        euler1d::Vec3 Qavg = cell_average_Q(grid, i, /*t=*/0.0, a, u0, p0, par);
        U.row(i) = Qavg.transpose();
    }
    return U;
}

inline std::vector<double> exact_rho_cell_averages(const fv1d::Grid1D& grid,
                                                   double t,
                                                   double a, double u0)
{
    std::vector<double> xi, w;
    gauss_legendre_4(xi, w);

    std::vector<double> rhoavg(grid.N, 0.0);
    const double PI = 3.14159265358979323846;
    const double xmin = grid.xmin, xmax = grid.xmax;
    const double L = xmax - xmin;

    for (int i = 0; i < grid.N; ++i) {
        const double xc = grid.xc[i];
        double s = 0.0;
        for (size_t q = 0; q < xi.size(); ++q) {
            const double xq = xc + 0.5 * grid.dx * xi[q];
            double xshift = wrap_periodic(xq - u0 * t, xmin, xmax);
            double rho = 1.0 + a * std::sin(2.0 * PI * (xshift - xmin) / L);
            s += w[q] * rho;
        }
        rhoavg[i] = 0.5 * s;
    }
    return rhoavg;
}

// convergence test
// domain [-1,1], periodic; u=1, p=1
// rho(x,t) = 2 + sin^4(pi*(x - t))
inline euler1d::Vec3 pdf_convergence_primitive(double x, double t,
                                               double xmin, double xmax)
{
    const double PI = 3.14159265358979323846;
    // u=1
    double xshift = wrap_periodic(x - 1.0 * t, xmin, xmax);
    double s = std::sin(PI * xshift);
    double rho = 2.0 + std::pow(s, 4.0);
    double u = 1.0;
    double p = 1.0;
    euler1d::Vec3 W; W << rho, u, p;
    return W;
}

inline euler1d::Vec3 cell_average_Q_pdf(const fv1d::Grid1D& grid, int i, double t,
                                        const euler1d::Params& par)
{
    std::vector<double> xi, w;
    gauss_legendre_4(xi, w);

    const double xc = grid.xc[i];
    const double dx = grid.dx;

    euler1d::Vec3 Qavg = euler1d::Vec3::Zero();

    for (size_t q = 0; q < xi.size(); ++q) {
        const double xq = xc + 0.5 * dx * xi[q];
        euler1d::Vec3 Wq = pdf_convergence_primitive(xq, t, grid.xmin, grid.xmax);
        euler1d::Vec3 Qq = euler1d::primToCons(Wq, par);
        Qavg += w[q] * Qq;
    }
    Qavg *= 0.5;
    return Qavg;
}

inline Eigen::MatrixXd init_pdf_cell_averages(const fv1d::Grid1D& grid,
                                              const euler1d::Params& par)
{
    Eigen::MatrixXd U(grid.N, 3);
    for (int i = 0; i < grid.N; ++i) {
        euler1d::Vec3 Qavg = cell_average_Q_pdf(grid, i, /*t=*/0.0, par);
        U.row(i) = Qavg.transpose();
    }
    return U;
}

inline std::vector<double> exact_rho_cell_averages_pdf(const fv1d::Grid1D& grid,
                                                       double t)
{
    std::vector<double> xi, w;
    gauss_legendre_4(xi, w);

    std::vector<double> rhoavg(grid.N, 0.0);
    const double PI = 3.14159265358979323846;

    for (int i = 0; i < grid.N; ++i) {
        const double xc = grid.xc[i];
        double s = 0.0;
        for (size_t q = 0; q < xi.size(); ++q) {
            const double xq = xc + 0.5 * grid.dx * xi[q];
            double xshift = wrap_periodic(xq - 1.0 * t, grid.xmin, grid.xmax); // u=1
            double ss = std::sin(PI * xshift);
            double rho = 2.0 + std::pow(ss, 4.0);
            s += w[q] * rho;
        }
        rhoavg[i] = 0.5 * s;
    }
    return rhoavg;
}


// Norms + drivers
struct Norms {
    double L1 = 0.0;
    double L2 = 0.0;
    double Linf = 0.0;
};

inline Norms rho_error_norms(const Eigen::MatrixXd& U,
                             const fv1d::Grid1D& grid,
                             const std::vector<double>& rho_exact_avg)
{
    if (U.rows() != grid.N || U.cols() != 3) throw std::runtime_error("rho_error_norms: U shape mismatch");
    if ((int)rho_exact_avg.size() != grid.N) throw std::runtime_error("rho_error_norms: exact size mismatch");

    Norms n;
    double s1 = 0.0, s2 = 0.0, sInf = 0.0;

    for (int i = 0; i < grid.N; ++i) {
        const double e = U(i,0) - rho_exact_avg[i]; // rho average
        const double ae = std::abs(e);
        s1 += ae * grid.dx;
        s2 += e * e * grid.dx;
        sInf = std::max(sInf, ae);
    }
    n.L1 = s1;
    n.L2 = std::sqrt(s2);
    n.Linf = sInf;
    return n;
}

// MUSCL evolution
// Can override limiter here.
inline Eigen::MatrixXd evolve_MUSCL(const Eigen::MatrixXd& U0,
                                   const fv1d::Grid1D& grid,
                                   int Nghost,
                                   const euler1d::Params& par,
                                   double CFL,
                                   double Tfinal)
{
    Eigen::MatrixXd U = U0;
    Eigen::MatrixXd Ubc(grid.N + 2*Nghost, 3);
    Eigen::MatrixXd Unp1;

    double t = 0.0;
    while (t < Tfinal) {
        fv1d::insert_interior_into_ghosted(U, Ubc, Nghost);
        fv1d::apply_periodic_bc(Ubc, grid.N, Nghost);

        double dt = compute_dt_cfl(U, grid.dx, CFL, par);
        if (t + dt > Tfinal) dt = Tfinal - t;

        // For smooth convergence we want Limiter::None.
        muscl1d::step(Ubc, grid.N, Nghost, dt, grid.dx, par, Unp1,
                      muscl1d::Limiter::None, muscl1d::BcType::Periodic,
                      muscl1d::FluxType::GodunovExact);

        U.swap(Unp1);
        t += dt;
    }
    return U;
}

inline Eigen::MatrixXd evolve_ADER3(const Eigen::MatrixXd& U0,
                                   const fv1d::Grid1D& grid,
                                   int Nghost,
                                   const euler1d::Params& par,
                                   double CFL,
                                   double Tfinal)
{
    WENO1d weno(/*PolyDegree=*/2, grid.dx);

    Eigen::MatrixXd U = U0;
    Eigen::MatrixXd Ubc(grid.N + 2*Nghost, 3);
    Eigen::MatrixXd Unp1(grid.N, 3);

    double t = 0.0;
    while (t < Tfinal) {
        fv1d::insert_interior_into_ghosted(U, Ubc, Nghost);
        fv1d::apply_periodic_bc(Ubc, grid.N, Nghost);

        double dt = compute_dt_cfl(U, grid.dx, CFL, par);
        if (t + dt > Tfinal) dt = Tfinal - t;

        std::vector<euler1d::Vec3> Fbar(grid.N + 1);
        for (int j = 0; j <= grid.N; ++j) {
            Fbar[j] = ADER3_interface_flux(weno, Ubc, j, Nghost, dt, par, /*alpha_diss=*/0.0);
        }

        for (int i = 0; i < grid.N; ++i) {
            euler1d::Vec3 Qi = U.row(i).transpose();
            euler1d::Vec3 Qnew = Qi - (dt / grid.dx) * (Fbar[i+1] - Fbar[i]);
            Unp1.row(i) = Qnew.transpose();
        }

        U.swap(Unp1);
        t += dt;
    }
    return U;
}

// timing wrapper, secs
template <class Func>
inline double time_seconds(Func&& f)
{
    auto t0 = std::chrono::steady_clock::now();
    f();
    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    return dt.count();
}

} // namespace step8