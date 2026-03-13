#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <stdexcept>

#include "Euler1D.h"

namespace exact_riemann {

using euler1d::Vec3;

// Inputs/outputs are primitive W = [rho, u, p]^T.
// Sample at position x, time t, discontinuity at x0.
struct ExactEulerRiemann {
    euler1d::Params par;

    explicit ExactEulerRiemann(const euler1d::Params& p) : par(p) {}

    static inline double sound_speed(double rho, double p, double gamma) {
        return std::sqrt(gamma * p / rho);
    }

    // f(p) and f'(p) for one side (Toro)
    static inline void pressure_function(double p,
                                         double rhoK, double uK, double pK,
                                         double aK, double gamma,
                                         double& f, double& df)
    {
        if (p > pK) {
            // Shock
            const double A = 2.0 / ((gamma + 1.0) * rhoK);
            const double B = (gamma - 1.0) / (gamma + 1.0) * pK;
            const double sq = std::sqrt(A / (p + B));
            f  = (p - pK) * sq;
            df = sq * (1.0 - 0.5 * (p - pK) / (p + B));
        } else {
            // Rarefaction
            const double pr = p / pK;
            const double g1 = (gamma - 1.0) / (2.0 * gamma);
            f  = (2.0 * aK / (gamma - 1.0)) * (std::pow(pr, g1) - 1.0);
            df = (1.0 / (rhoK * aK)) * std::pow(pr, -(gamma + 1.0) / (2.0 * gamma));
        }
    }

    // Pressure guess (Toro)
    static inline double guess_pressure(double rhoL, double uL, double pL, double aL,
                                        double rhoR, double uR, double pR, double aR,
                                        double gamma)
    {
        const double pPV = 0.5 * (pL + pR) - 0.125 * (uR - uL) * (rhoL + rhoR) * (aL + aR);
        const double pmin = std::min(pL, pR);
        const double pmax = std::max(pL, pR);
        const double qmax = pmax / pmin;

        if (qmax <= 2.0 && pPV >= pmin && pPV <= pmax) {
            // PVRS
            return std::max(1e-14, pPV);
        }

        if (pPV < pmin) {
            // Two-rarefaction approximation
            const double g1 = (gamma - 1.0) / (2.0 * gamma);
            const double pTR = std::pow(
                (aL + aR - 0.5 * (gamma - 1.0) * (uR - uL)) /
                (aL * std::pow(pL, -g1) + aR * std::pow(pR, -g1)),
                1.0 / g1
            );
            return std::max(1e-14, pTR);
        }

        // Two-shock approximation
        const double AL = 2.0 / ((gamma + 1.0) * rhoL);
        const double BL = (gamma - 1.0) / (gamma + 1.0) * pL;
        const double AR = 2.0 / ((gamma + 1.0) * rhoR);
        const double BR = (gamma - 1.0) / (gamma + 1.0) * pR;

        const double gL = std::sqrt(AL / (pPV + BL));
        const double gR = std::sqrt(AR / (pPV + BR));
        const double pTS = (gL * pL + gR * pR - (uR - uL)) / (gL + gR);
        return std::max(1e-14, pTS);
    }

    // Solve for p* and u*
    inline void solve_star(const Vec3& WL, const Vec3& WR, double& pstar, double& ustar) const
    {
        const double rhoL = WL(0), uL = WL(1), pL = WL(2);
        const double rhoR = WR(0), uR = WR(1), pR = WR(2);
        const double g = par.gamma;

        if (par.strict_checks) {
            if (!(rhoL > 0 && pL > 0 && rhoR > 0 && pR > 0))
                throw std::runtime_error("ExactEulerRiemann: nonphysical input states");
        }

        const double aL = sound_speed(rhoL, pL, g);
        const double aR = sound_speed(rhoR, pR, g);

        double p = guess_pressure(rhoL, uL, pL, aL, rhoR, uR, pR, aR, g);
        p = std::max(p, par.p_floor);

        // Newton iterations
        for (int it = 0; it < 30; ++it) {
            double fL, dfL, fR, dfR;
            pressure_function(p, rhoL, uL, pL, aL, g, fL, dfL);
            pressure_function(p, rhoR, uR, pR, aR, g, fR, dfR);

            const double f  = fL + fR + (uR - uL);
            const double df = dfL + dfR;

            double p_new = p - f / df;
            p_new = std::max(p_new, par.p_floor);

            const double rel = std::abs(p_new - p) / (0.5 * (p_new + p) + 1e-16);
            p = p_new;
            if (rel < 1e-10) break;
        }

        pstar = p;

        // u* from Toro: u* = 0.5*(uL+uR + fR - fL) evaluated at p*
        double fL, dfL, fR, dfR;
        pressure_function(pstar, rhoL, uL, pL, aL, g, fL, dfL);
        pressure_function(pstar, rhoR, uR, pR, aR, g, fR, dfR);
        ustar = 0.5 * (uL + uR + fR - fL);
    }

    // Sample solution W(x,t) at given xi=(x-x0)/t
    inline Vec3 sample(const Vec3& WL, const Vec3& WR, double x, double t, double x0) const
    {
        if (t <= 0.0) {
            return (x < x0) ? WL : WR;
        }

        const double rhoL = WL(0), uL = WL(1), pL = WL(2);
        const double rhoR = WR(0), uR = WR(1), pR = WR(2);
        const double g = par.gamma;

        const double aL = sound_speed(rhoL, pL, g);
        const double aR = sound_speed(rhoR, pR, g);

        double pstar, ustar;
        solve_star(WL, WR, pstar, ustar);

        const double xi = (x - x0) / t;

        // Decide whether xi is left or right of contact (ustar)
        if (xi <= ustar) {
            // LEFT SIDE
            if (pstar > pL) {
                // Left shock
                const double SL = uL - aL * std::sqrt((g + 1.0) / (2.0 * g) * (pstar / pL) + (g - 1.0) / (2.0 * g));
                if (xi <= SL) return WL;

                const double ratio = pstar / pL;
                const double rho_star_L = rhoL * ( (ratio + (g - 1.0)/(g + 1.0)) / ( (g - 1.0)/(g + 1.0)*ratio + 1.0 ) );
                Vec3 W; W << rho_star_L, ustar, pstar;
                return W;
            } else {
                // Left rarefaction
                const double SHL = uL - aL;
                const double a_star_L = aL * std::pow(pstar / pL, (g - 1.0) / (2.0 * g));
                const double STL = ustar - a_star_L;

                if (xi <= SHL) return WL;
                if (xi >= STL) {
                    const double rho_star_L = rhoL * std::pow(pstar / pL, 1.0 / g);
                    Vec3 W; W << rho_star_L, ustar, pstar;
                    return W;
                }

                // Inside left fan
                const double u = (2.0 / (g + 1.0)) * (aL + 0.5 * (g - 1.0) * uL + xi);
                const double a = (2.0 / (g + 1.0)) * (aL + 0.5 * (g - 1.0) * (uL - xi));
                const double rho = rhoL * std::pow(a / aL, 2.0 / (g - 1.0));
                const double p   = pL   * std::pow(a / aL, 2.0 * g / (g - 1.0));
                Vec3 W; W << rho, u, p;
                return W;
            }
        } else {
            // RIGHT SIDE
            if (pstar > pR) {
                // Right shock
                const double SR = uR + aR * std::sqrt((g + 1.0) / (2.0 * g) * (pstar / pR) + (g - 1.0) / (2.0 * g));
                if (xi >= SR) return WR;

                const double ratio = pstar / pR;
                const double rho_star_R = rhoR * ( (ratio + (g - 1.0)/(g + 1.0)) / ( (g - 1.0)/(g + 1.0)*ratio + 1.0 ) );
                Vec3 W; W << rho_star_R, ustar, pstar;
                return W;
            } else {
                // Right rarefaction
                const double SHR = uR + aR;
                const double a_star_R = aR * std::pow(pstar / pR, (g - 1.0) / (2.0 * g));
                const double STR = ustar + a_star_R;

                if (xi >= SHR) return WR;
                if (xi <= STR) {
                    const double rho_star_R = rhoR * std::pow(pstar / pR, 1.0 / g);
                    Vec3 W; W << rho_star_R, ustar, pstar;
                    return W;
                }

                // Inside right fan
                const double u = (2.0 / (g + 1.0)) * (-aR + 0.5 * (g - 1.0) * uR + xi);
                const double a = (2.0 / (g + 1.0)) * ( aR - 0.5 * (g - 1.0) * (uR - xi) );
                const double rho = rhoR * std::pow(a / aR, 2.0 / (g - 1.0));
                const double p   = pR   * std::pow(a / aR, 2.0 * g / (g - 1.0));
                Vec3 W; W << rho, u, p;
                return W;
            }
        }
    }
};

} // namespace exact_riemann