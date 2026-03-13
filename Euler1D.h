#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <string>

namespace euler1d {

using Vec3 = Eigen::Vector3d;

struct Params {
    double gamma = 1.4;
    double rho_floor = 1e-12;
    double p_floor   = 1e-12;
    bool strict_checks = true;
};

inline double clamp_min(double x, double xmin) {
    return (x < xmin) ? xmin : x;
}

inline void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

// Primitive W = [rho, u, p]^T -> Conserved Q = [rho, rho*u, E]^T
inline Vec3 primToCons(const Vec3& W, const Params& par) {
    const double rho = W(0);
    const double u   = W(1);
    const double p   = W(2);

    if (par.strict_checks) {
        require(rho > 0.0, "primToCons: rho <= 0");
        require(p   > 0.0, "primToCons: p <= 0");
    }

    const double rho_c = par.strict_checks ? rho : clamp_min(rho, par.rho_floor);
    const double p_c   = par.strict_checks ? p   : clamp_min(p,   par.p_floor);

    const double mom = rho_c * u;
    const double E   = p_c / (par.gamma - 1.0) + 0.5 * rho_c * u * u;

    return Vec3(rho_c, mom, E);
}

// Conserved Q = [rho, rho*u, E]^T -> Primitive W = [rho, u, p]^T
inline Vec3 consToPrim(const Vec3& Q, const Params& par) {
    const double rho = Q(0);
    const double mom = Q(1);
    const double E   = Q(2);

    if (par.strict_checks) {
        require(rho > 0.0, "consToPrim: rho <= 0");
    }

    const double rho_c = par.strict_checks ? rho : clamp_min(rho, par.rho_floor);
    const double u     = mom / rho_c;

    // p = (gamma-1) * (E - 0.5*rho*u^2)
    const double kinetic = 0.5 * rho_c * u * u;
    double p = (par.gamma - 1.0) * (E - kinetic);

    if (par.strict_checks) {
        require(p > 0.0, "consToPrim: p <= 0 (non-physical state)");
    } else {
        p = clamp_min(p, par.p_floor);
    }

    return Vec3(rho_c, u, p);
}


// Physics: pressure, flux, sound speed, wave speed
inline double pressureFromCons(const Vec3& Q, const Params& par) {
    // Uses the same formula as consToPrim, but returns only p.
    const double rho = Q(0);
    const double mom = Q(1);
    const double E   = Q(2);

    const double rho_c = par.strict_checks ? rho : clamp_min(rho, par.rho_floor);
    if (par.strict_checks) require(rho_c > 0.0, "pressureFromCons: rho <= 0");

    const double u = mom / rho_c;
    const double kinetic = 0.5 * rho_c * u * u;
    double p = (par.gamma - 1.0) * (E - kinetic);

    if (par.strict_checks) {
        require(p > 0.0, "pressureFromCons: p <= 0");
    } else {
        p = clamp_min(p, par.p_floor);
    }
    return p;
}

inline double velocityFromCons(const Vec3& Q, const Params& par) {
    const double rho = Q(0);
    const double mom = Q(1);
    const double rho_c = par.strict_checks ? rho : clamp_min(rho, par.rho_floor);
    if (par.strict_checks) require(rho_c > 0.0, "velocityFromCons: rho <= 0");
    return mom / rho_c;
}

// Euler flux F(Q) = [rho*u, rho*u^2 + p, u(E+p)]^T
inline Vec3 flux(const Vec3& Q, const Params& par) {
    const double rho = Q(0);
    const double mom = Q(1);
    const double E   = Q(2);

    const double rho_c = par.strict_checks ? rho : clamp_min(rho, par.rho_floor);
    if (par.strict_checks) require(rho_c > 0.0, "flux: rho <= 0");

    const double u = mom / rho_c;
    const double p = pressureFromCons(Q, par);

    Vec3 F;
    F(0) = mom;
    F(1) = mom * u + p;
    F(2) = u * (E + p);
    return F;
}

// Sound speed c = sqrt(gamma*p/rho) given primitive W=[rho,u,p]
inline double soundSpeedFromPrim(const Vec3& W, const Params& par) {
    const double rho = W(0);
    const double p   = W(2);

    const double rho_c = par.strict_checks ? rho : clamp_min(rho, par.rho_floor);
    const double p_c   = par.strict_checks ? p   : clamp_min(p,   par.p_floor);

    if (par.strict_checks) {
        require(rho_c > 0.0, "soundSpeedFromPrim: rho <= 0");
        require(p_c   > 0.0, "soundSpeedFromPrim: p <= 0");
    }

    return std::sqrt(par.gamma * p_c / rho_c);
}

// Max wave speed for CFL: |u| + c, computed from conserved Q.
inline double maxWaveSpeed(const Vec3& Q, const Params& par) {
    const Vec3 W = consToPrim(Q, par);
    const double u = W(1);
    const double c = soundSpeedFromPrim(W, par);
    return std::abs(u) + c;
}

} // namespace euler1d
