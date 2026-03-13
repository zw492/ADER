#pragma once
#include "Euler1D.h"
#include "WenoAdapter.h"
#include "CK_Euler.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "ExactRiemannEuler1D.h"

// ADER3 following the TT GRP
namespace ader3_ttgrp {

using Vec3 = euler1d::Vec3;

struct Eigensystem {
    Eigen::Matrix3d R;
    Eigen::FullPivLU<Eigen::Matrix3d> lu;
    Eigen::Vector3d lam;
    double a = 0.0;
};

inline Eigensystem eigensystem_from_state(const Vec3& Q,
                                          const euler1d::Params& par)
{
    Eigensystem sys;
    const double g = par.gamma;
    const Vec3 W = euler1d::consToPrim(Q, par);
    const double rho = W(0), u = W(1), p = W(2);
    const double c = std::sqrt(g * p / rho);
    const double H = (Q(2) + p) / rho;

    sys.a = c;
    sys.lam << (u - c), u, (u + c);

    sys.R.col(0) << 1.0, (u - c), (H - u * c);
    sys.R.col(1) << 1.0, u,       0.5 * u * u;
    sys.R.col(2) << 1.0, (u + c), (H + u * c);

    sys.lu.compute(sys.R);
    if (!sys.lu.isInvertible())
        throw std::runtime_error("eigensystem_from_state: eigenvector matrix not invertible");

    return sys;
}

inline void upwind_characteristic_derivatives(
    const Vec3& QxL,  const Vec3& QxR,
    const Vec3& QxxL, const Vec3& QxxR,
    const Eigensystem& sys,
    Vec3& Qx0, Vec3& Qxx0)
{
    const Eigen::Vector3d wL  = sys.lu.solve(QxL);
    const Eigen::Vector3d wR  = sys.lu.solve(QxR);
    const Eigen::Vector3d wwL = sys.lu.solve(QxxL);
    const Eigen::Vector3d wwR = sys.lu.solve(QxxR);

    Eigen::Vector3d w0, ww0;
    for (int k = 0; k < 3; ++k) {
        if (sys.lam(k) >= 0.0) {
            w0(k)  = wL(k);
            ww0(k) = wwL(k);
        } else {
            w0(k)  = wR(k);
            ww0(k) = wwR(k);
        }
    }

    Qx0  = sys.R * w0;
    Qxx0 = sys.R * ww0;
}

// Takes the pre-converted VectOfVectDouble to avoid repeated conversion.
inline euler1d::Vec3 compute_flux(
        const WENO1d& weno,
        const VectOfVectDouble& Ubc_vov,
        int j, int Nghost,
        double dt,
        const euler1d::Params& par,
        double alpha_diss)
{
    // Step 1: WENO reconstruction
    VectOfVectDouble U_WenoL, U_WenoR;
    weno.WENO_reconstructionForFluxEvaluation(Ubc_vov, j, U_WenoL, U_WenoR, Nghost);

    // U_WenoL[d][n]: d = derivative order, n = variable
    const Vec3 QL   = { U_WenoL[0][0], U_WenoL[0][1], U_WenoL[0][2] };
    const Vec3 QxL  = { U_WenoL[1][0], U_WenoL[1][1], U_WenoL[1][2] };
    const Vec3 QxxL = { U_WenoL[2][0], U_WenoL[2][1], U_WenoL[2][2] };
    const Vec3 QR   = { U_WenoR[0][0], U_WenoR[0][1], U_WenoR[0][2] };
    const Vec3 QxR  = { U_WenoR[1][0], U_WenoR[1][1], U_WenoR[1][2] };
    const Vec3 QxxR = { U_WenoR[2][0], U_WenoR[2][1], U_WenoR[2][2] };

    // Step 2: Exact Riemann solver -> Godunov state Q0 = G(0)
    exact_riemann::ExactEulerRiemann ex(par);
    const Vec3 WL = euler1d::consToPrim(QL, par);
    const Vec3 WR = euler1d::consToPrim(QR, par);
    const Vec3 W0 = ex.sample(WL, WR, 0.0, 1e-12, 0.0);
    const Vec3 Q0 = euler1d::primToCons(W0, par);

    // Step 3: Eigensystem of A(Q0)
    const auto sys0 = eigensystem_from_state(Q0, par);

    // Step 4: Upwind spatial derivatives in characteristic basis of A(Q0)
    Vec3 Qx0, Qxx0;
    upwind_characteristic_derivatives(QxL, QxR, QxxL, QxxR, sys0, Qx0, Qxx0);

    // Step 5: CK -> Qt, Qtt
    Vec3 Qt, Qtt;
    Eigen::Matrix3d A, At;
    CK_Euler_qt_qtt(Q0, Qx0, Qxx0, par, Qt, Qtt, &A, &At);

    // Step 6: 2-point Gauss-Legendre quadrature of F(J(tau)) over [0, dt]
    // Nodes on [0,1]: (1 ± 1/√3)/2, weights: 1/2 each
    const double s  = 1.0 / std::sqrt(3.0);
    const double g1 = 0.5 * (1.0 - s) * dt;
    const double g2 = 0.5 * (1.0 + s) * dt;
    const Vec3 J1 = Q0 + g1 * Qt + (0.5 * g1 * g1) * Qtt;
    const Vec3 J2 = Q0 + g2 * Qt + (0.5 * g2 * g2) * Qtt;
    Vec3 Fbar = 0.5 * euler1d::flux(J1, par) + 0.5 * euler1d::flux(J2, par);

    // Step 7: Optional Rusanov dissipation (NOT in use for ANY tasks)
    if (alpha_diss > 0.0) {
        const double amax = std::max(euler1d::maxWaveSpeed(QL, par),
                                     euler1d::maxWaveSpeed(QR, par));
        Fbar -= 0.5 * alpha_diss * amax * (QR - QL);
    }

    return Fbar;
}

} // namespace ader3_ttgrp



inline euler1d::Vec3 ADER3_interface_flux(
        const WENO1d& weno,
        const VectOfVectDouble& Ubc_vov,
        int j, int Nghost,
        double dt,
        const euler1d::Params& par,
        double alpha_diss = 0.0)
{
    return ader3_ttgrp::compute_flux(weno, Ubc_vov, j, Nghost, dt, par, alpha_diss);
}


inline euler1d::Vec3 ADER3_interface_flux(
        const WENO1d& weno,
        const Eigen::MatrixXd& Ubc,
        int j, int Nghost,
        double dt,
        const euler1d::Params& par,
        double alpha_diss = 0.0)
{
    const VectOfVectDouble Ubc_vov = weno_adapter::eigen_to_vov(Ubc);
    return ader3_ttgrp::compute_flux(weno, Ubc_vov, j, Nghost, dt, par, alpha_diss);
}