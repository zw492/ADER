#pragma once
#include "Euler1D.h"
#include <algorithm>
#include <cmath>
#include "ExactRiemannEuler1D.h"

inline euler1d::Vec3 rusanov_flux(const euler1d::Vec3& QL,
                                  const euler1d::Vec3& QR,
                                  const euler1d::Params& par)
{
    const euler1d::Vec3 FL = euler1d::flux(QL, par);
    const euler1d::Vec3 FR = euler1d::flux(QR, par);

    const double aL = euler1d::maxWaveSpeed(QL, par);
    const double aR = euler1d::maxWaveSpeed(QR, par);
    const double a  = std::max(aL, aR);

    return 0.5 * (FL + FR) - 0.5 * a * (QR - QL);
}

// Godunov flux using exact Riemann solution (Toro)
inline euler1d::Vec3 godunov_exact_flux(const euler1d::Vec3& QL,
                                        const euler1d::Vec3& QR,
                                        const euler1d::Params& par)
{
    const euler1d::Vec3 WL = euler1d::consToPrim(QL, par);
    const euler1d::Vec3 WR = euler1d::consToPrim(QR, par);

    exact_riemann::ExactEulerRiemann ex(par);

    // sample at xi=0 (use tiny t to avoid division-by-zero in sampling logic)
    const double t_eps = 1e-12;
    const euler1d::Vec3 W0 = ex.sample(WL, WR, /*x=*/0.0, /*t=*/t_eps, /*x0=*/0.0);
    const euler1d::Vec3 Q0 = euler1d::primToCons(W0, par);

    return euler1d::flux(Q0, par);
}
