#pragma once
#include <Eigen/Dense>
#include <stdexcept>
#include "Euler1D.h"

// CK for 1D Euler, ideal gas.
// Inputs: Q, Qx, Qxx in conserved variables.
// Outputs: Qt, Qtt in conserved variables.
// Q = [rho, m, E]^T, where m=rho*u.
// p = (gamma-1)*(E - 0.5*m^2/rho).
// CK identities used (3rd order time requires up to Qtt):
//   Qt  = - A(Q) * Qx
//   Ax  = dA/dQ * Qx
//   At  = dA/dQ * Qt
//   Qxt = -(Ax Qx + A Qxx)
//   Qtt = -(At Qx + A Qxt)

inline void CK_Euler_qt_qtt(
    const euler1d::Vec3& Q,
    const euler1d::Vec3& Qx,
    const euler1d::Vec3& Qxx,
    const euler1d::Params& par,
    euler1d::Vec3& Qt,
    euler1d::Vec3& Qtt,
    Eigen::Matrix3d* A_out  = nullptr,
    Eigen::Matrix3d* At_out = nullptr)
{
    const double g   = par.gamma;
    const double rho = Q(0);
    const double m   = Q(1);
    const double E   = Q(2);

    if (par.strict_checks) {
        if (!(rho > 0.0)) throw std::runtime_error("CK_Euler_qt_qtt: rho <= 0");
        const double p_chk = (g - 1.0) * (E - 0.5 * m * m / rho);
        if (!(p_chk > 0.0)) throw std::runtime_error("CK_Euler_qt_qtt: p <= 0");
    } else {
        if (!(rho > par.rho_floor))
            throw std::runtime_error("CK_Euler_qt_qtt: rho too small even with floors");
    }

    // Euler flux Jacobian A = dF/dQ (conserved variables Q=[rho, m, E])
    // F1 = m
    // F2 = m^2/rho + (g-1)*(E - m^2/(2*rho)) = (3-g)/2 * m^2/rho + (g-1)*E
    // F3 = m/rho * (g*E - (g-1)*m^2/(2*rho))
    // A = dF/dQ (row i = dFi/dQ):

    Eigen::Matrix3d A;
    A <<
        0.0,  1.0,  0.0,
        (m*m)*(g - 3.0)/(2.0*rho*rho), m*(3.0 - g)/rho, (g - 1.0),
        m*(-E*g*rho + g*m*m - m*m)/(rho*rho*rho),
        (2.0*E*g*rho - 3.0*g*m*m + 3.0*m*m)/(2.0*rho*rho),
        g*m/rho;


    // Partial derivatives of A wrt (rho, m, E): dA_drho, dA_dm, dA_dE
    Eigen::Matrix3d dA_drho, dA_dm, dA_dE;

    dA_drho <<
        0.0,  0.0,  0.0,
        (m*m)*(3.0 - g)/(rho*rho*rho),   m*(g - 3.0)/(rho*rho),   0.0,
        m*(2.0*E*g*rho - 3.0*g*m*m + 3.0*m*m)/(rho*rho*rho*rho),
        (-E*g*rho + 3.0*g*m*m - 3.0*m*m)/(rho*rho*rho),
        -g*m/(rho*rho);

    dA_dm <<
        0.0,  0.0,  0.0,
        m*(g - 3.0)/(rho*rho),   (3.0 - g)/rho,   0.0,
        (-E*g*rho + 3.0*g*m*m - 3.0*m*m)/(rho*rho*rho),
        3.0*m*(1.0 - g)/(rho*rho),
        g/rho;

    dA_dE <<
        0.0,  0.0,  0.0,
        0.0,  0.0,  0.0,
        -g*m/(rho*rho),   g/rho,   0.0;


    // apply_dA: contract (dA/dQ) with a vector dQ  ->  3x3 matrix

    auto apply_dA = [&](const euler1d::Vec3& dQ) -> Eigen::Matrix3d {
        return dA_drho * dQ(0) + dA_dm * dQ(1) + dA_dE * dQ(2);
    };

    Qt = -(A * Qx);

    Eigen::Matrix3d Ax = apply_dA(Qx);
    Eigen::Matrix3d At = apply_dA(Qt);

    euler1d::Vec3 Qxt = -(Ax * Qx + A * Qxx);

    Qtt = -(At * Qx + A * Qxt);

    if (A_out)  *A_out  = A;
    if (At_out) *At_out = At;
}