#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <stdexcept>
#include "Euler1D.h"

// CFL timestep based on max_i (|u_i| + c_i) from conserved states Q_i
// U: N x 3 matrix of conserved variables [rho, rho*u, E]
inline double compute_dt_cfl(const Eigen::MatrixXd& U,
                             double dx,
                             double CFL,
                             const euler1d::Params& par)
{
    if (U.cols() != 3) throw std::runtime_error("compute_dt_cfl: U must have 3 columns");
    if (dx <= 0.0) throw std::runtime_error("compute_dt_cfl: dx must be > 0");
    if (CFL <= 0.0) throw std::runtime_error("compute_dt_cfl: CFL must be > 0");

    double amax = 0.0;
    for (int i = 0; i < U.rows(); ++i) {
        euler1d::Vec3 Qi = U.row(i).transpose();
        amax = std::max(amax, euler1d::maxWaveSpeed(Qi, par));
    }
    if (!(amax > 0.0)) throw std::runtime_error("compute_dt_cfl: amax <= 0 (non-physical?)");
    return CFL * dx / amax;
}
