#pragma once
#include <Eigen/Dense>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <vector>

#include "Euler1D.h"
#include "RiemannFlux.h"


namespace muscl1d {
enum class FluxType { Rusanov, GodunovExact };

// Limiter choices
enum class Limiter {
    None,
    Minmod,
    MC,
    VanLeer
};

// Boundary type for ghost fill at the half step
enum class BcType {
    Periodic,
    Outflow
};

// scalar minmod
inline double minmod(double a, double b)
{
    if (a * b <= 0.0) return 0.0;
    return (std::abs(a) < std::abs(b)) ? a : b;
}

// MC limiter: minmod( 0.5(a+b), 2a, 2b )
inline double mc_limited(double dm, double dp)
{
    const double s = 0.5 * (dm + dp);
    return minmod(s, minmod(2.0 * dm, 2.0 * dp));
}

// Van Leer limiter: 2|dm||dp| / (|dm|+|dp|), sign of (dm+dp)
inline double van_leer_limited(double dm, double dp)
{
    if (dm * dp <= 0.0) return 0.0;
    const double adm = std::abs(dm), adp = std::abs(dp);
    const double sign = ((dm + dp) > 0.0) ? 1.0 : -1.0;
    return sign * 2.0 * adm * adp / (adm + adp);
}

// Select slope based on limiter
inline double limited_slope(double dm, double dp, Limiter lim)
{
    switch (lim) {
        case Limiter::None:
            return 0.5 * (dm + dp);
        case Limiter::Minmod:
            return minmod(dm, dp);
        case Limiter::VanLeer:
            return van_leer_limited(dm, dp);
        case Limiter::MC:
        default:
            return mc_limited(dm, dp);
    }
}


// Convert conserved matrix -> primitive matrix (row-wise)
inline void cons_to_prim_matrix(const Eigen::MatrixXd& U,
                                Eigen::MatrixXd& W,
                                const euler1d::Params& par)
{
    if (U.cols() != 3) throw std::runtime_error("cons_to_prim_matrix: U must have 3 columns");
    W.resize(U.rows(), 3);
    for (int i = 0; i < U.rows(); ++i) {
        euler1d::Vec3 Qi = U.row(i).transpose();
        euler1d::Vec3 Wi = euler1d::consToPrim(Qi, par);
        W.row(i) = Wi.transpose();
    }
}


// Fill ghosts for a ghosted conserved matrix Ubc (size N+2*Nghost x 3)
// Convention: interior i=0......N-1 stored at row (Nghost+i).
inline void apply_bc(Eigen::MatrixXd& Ubc, int N, int Nghost, BcType bc)
{
    if (Ubc.rows() != N + 2 * Nghost || Ubc.cols() != 3)
        throw std::runtime_error("apply_bc: size mismatch");

    const int first_int_row = Nghost;
    const int last_int_row  = Nghost + N - 1;

    if (bc == BcType::Outflow) {
        for (int g = 0; g < Nghost; ++g) {
            Ubc.row(g) = Ubc.row(first_int_row);
            Ubc.row(Nghost + N + g) = Ubc.row(last_int_row);
        }
        return;
    }

    // Periodic
    for (int g = 0; g < Nghost; ++g) {
        // left ghosts from last Nghost interior
        const int src_i = N - Nghost + g; // interior index
        Ubc.row(g) = Ubc.row(Nghost + src_i);

        // right ghosts from first Nghost interior
        const int dst_row = Nghost + N + g;
        const int src_j = g;             // interior index
        Ubc.row(dst_row) = Ubc.row(Nghost + src_j);
    }
}


// One MUSCL–Hancock update on interior cells.
// Inputs:
// Ubc: (N+2*Nghost) x 3 ghosted conserved array, BCs already applied at time n
// dt, dx
// Output:
// Unp1: N x 3 updated interior conserved array

// Notes:
// Compute half-time cell averages Qhalf (interior), then build Qhalf_bc with
// consistent BC filling (Periodic/Outflow) before reconstructing half-time interface states.

inline void step(const Eigen::MatrixXd& Ubc,
                 int N, int Nghost,
                 double dt, double dx,
                 const euler1d::Params& par,
                 Eigen::MatrixXd& Unp1,
                 Limiter limiter = Limiter::MC,
                 BcType bcType   = BcType::Periodic,
                 FluxType fluxType = FluxType::Rusanov)
                  //We are in fact using Exact Godunov for all tasks; even if it is Rusanov here.
{
    if (Ubc.rows() != N + 2 * Nghost || Ubc.cols() != 3)
        throw std::runtime_error("muscl1d::step: Ubc size mismatch");
    if (Nghost < 2)
        throw std::runtime_error("muscl1d::step: need Nghost >= 2 for MUSCL");

    // Convert to primitive everywhere at time n (including ghosts)
    Eigen::MatrixXd Wbc;
    cons_to_prim_matrix(Ubc, Wbc, par);

    // Slopes dW (primitive) for interior cells only: N x 3
    Eigen::MatrixXd dW(N, 3);
    dW.setZero();
    for (int i = 0; i < N; ++i) {
        const int gi = Nghost + i;
        for (int v = 0; v < 3; ++v) {
            const double dm = Wbc(gi, v) - Wbc(gi - 1, v);
            const double dp = Wbc(gi + 1, v) - Wbc(gi, v);
            dW(i, v) = limited_slope(dm, dp, limiter);
        }
    }

    // Reconstruct interface primitive states at time n
    // Interfaces j = 0..N (between cell j-1 and j)
    std::vector<euler1d::Vec3> QL_n(N + 1), QR_n(N + 1);
    for (int j = 0; j <= N; ++j) {
        const int iL = j - 1;
        const int iR = j;

        // cell-center primitives from Wbc
        // left cell center row in Wbc is (Nghost+iL), right is (Nghost+iR)
        euler1d::Vec3 WLc = Wbc.row(Nghost + iL).transpose();
        euler1d::Vec3 WRc = Wbc.row(Nghost + iR).transpose();
        euler1d::Vec3 dWL = euler1d::Vec3::Zero();
        euler1d::Vec3 dWR = euler1d::Vec3::Zero();

        auto wrap = [N](int k) {
            int r = k % N;
            if (r < 0) r += N;
            return r;
        };

        if (bcType == BcType::Periodic) {
            dWL = dW.row(wrap(iL)).transpose();
            dWR = dW.row(wrap(iR)).transpose();
        } else {
            if (0 <= iL && iL < N) dWL = dW.row(iL).transpose();
            if (0 <= iR && iR < N) dWR = dW.row(iR).transpose();
        }

        // W_L = W_iL + 0.5*dW_iL ; W_R = W_iR - 0.5*dW_iR
        const euler1d::Vec3 WL = WLc + 0.5 * dWL;
        const euler1d::Vec3 WR = WRc - 0.5 * dWR;

        QL_n[j] = euler1d::primToCons(WL, par);
        QR_n[j] = euler1d::primToCons(WR, par);
    }

    // Fluxes at time n
    std::vector<euler1d::Vec3> Fn(N + 1);
    for (int j = 0; j <= N; ++j) {
        auto num_flux = [&](const euler1d::Vec3& a, const euler1d::Vec3& b) {
            if (fluxType == FluxType::GodunovExact) return godunov_exact_flux(a, b, par);
            return rusanov_flux(a, b, par);
        };

        Fn[j] = num_flux(QL_n[j], QR_n[j]);
    }

    // Hancock predictor (half-step cell averages, interior only)
    Eigen::MatrixXd Qhalf(N, 3);
    for (int i = 0; i < N; ++i) {
        const euler1d::Vec3 Qi = Ubc.row(Nghost + i).transpose();
        const euler1d::Vec3 Qh = Qi - 0.5 * (dt / dx) * (Fn[i + 1] - Fn[i]);
        Qhalf.row(i) = Qh.transpose();
    }

    // Build ghosted half-step conserved array Qhalf_bc and apply BC consistently
    Eigen::MatrixXd Qhalf_bc(N + 2 * Nghost, 3);
    Qhalf_bc.setZero();
    Qhalf_bc.block(Nghost, 0, N, 3) = Qhalf;
    apply_bc(Qhalf_bc, N, Nghost, bcType);

    // Convert half-step to primitives (including ghosts)
    Eigen::MatrixXd Whalf_bc;
    cons_to_prim_matrix(Qhalf_bc, Whalf_bc, par);

    // Reconstruct interface states at half time using slopes from time n
    std::vector<euler1d::Vec3> QL_h(N + 1), QR_h(N + 1);
    for (int j = 0; j <= N; ++j) {
        const int iL = j - 1;
        const int iR = j;

        euler1d::Vec3 WLc = Whalf_bc.row(Nghost + iL).transpose();
        euler1d::Vec3 WRc = Whalf_bc.row(Nghost + iR).transpose();
        euler1d::Vec3 dWL = euler1d::Vec3::Zero();
        euler1d::Vec3 dWR = euler1d::Vec3::Zero();

        auto wrap = [N](int k) {
            int r = k % N;
            if (r < 0) r += N;
            return r;
        };

        if (bcType == BcType::Periodic) {
            dWL = dW.row(wrap(iL)).transpose();
            dWR = dW.row(wrap(iR)).transpose();
        } else {

    if (0 <= iL && iL < N) dWL = dW.row(iL).transpose();
    if (0 <= iR && iR < N) dWR = dW.row(iR).transpose();
}

        const euler1d::Vec3 WL = WLc + 0.5 * dWL;
        const euler1d::Vec3 WR = WRc - 0.5 * dWR;

        QL_h[j] = euler1d::primToCons(WL, par);
        QR_h[j] = euler1d::primToCons(WR, par);
    }

    // Fluxes at half time
    std::vector<euler1d::Vec3> Fh(N + 1);
    for (int j = 0; j <= N; ++j) {
        auto num_flux = [&](const euler1d::Vec3& a, const euler1d::Vec3& b) {
            if (fluxType == FluxType::GodunovExact) return godunov_exact_flux(a, b, par);
            return rusanov_flux(a, b, par);
        };
        Fh[j] = num_flux(QL_h[j], QR_h[j]);
    }

    // Final FV update
    Unp1.resize(N, 3);
    for (int i = 0; i < N; ++i) {
        const euler1d::Vec3 Qi = Ubc.row(Nghost + i).transpose();
        const euler1d::Vec3 Qnp1 = Qi - (dt / dx) * (Fh[i + 1] - Fh[i]);
        Unp1.row(i) = Qnp1.transpose();
    }
}

} // namespace muscl1d