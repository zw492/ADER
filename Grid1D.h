#pragma once
#include <Eigen/Dense>
#include <vector>
#include <stdexcept>
#include <string>

namespace fv1d {

struct Grid1D {
    double xmin = 0.0;
    double xmax = 1.0;
    int    N    = 0; 
    double dx   = 0.0;

    std::vector<double> xc; 

    Grid1D(double xmin_, double xmax_, int N_)
        : xmin(xmin_), xmax(xmax_), N(N_)
    {
        if (N <= 0) throw std::runtime_error("Grid1D: N must be > 0");
        dx = (xmax - xmin) / static_cast<double>(N);
        xc.resize(N);
        for (int i = 0; i < N; ++i) {
            xc[i] = xmin + (i + 0.5) * dx;
        }
    }
};

// Allocate interior and ghosted arrays
// U   : N x 3
// Ubc : (N + 2*Nghost) x 3
inline void allocate_state(int N, int Nghost, Eigen::MatrixXd& U, Eigen::MatrixXd& Ubc, int nvars = 3)
{
    if (N <= 0) throw std::runtime_error("allocate_state: N must be > 0");
    if (Nghost < 0) throw std::runtime_error("allocate_state: Nghost must be >= 0");
    if (nvars <= 0) throw std::runtime_error("allocate_state: nvars must be > 0");

    U.resize(N, nvars);
    U.setZero();

    Ubc.resize(N + 2 * Nghost, nvars);
    Ubc.setZero();
}

// Copy interior U into the interior region of Ubc
// Convention: interior cell i=0.....N-1 is stored at row (Nghost + i) in Ubc.
inline void insert_interior_into_ghosted(const Eigen::MatrixXd& U, Eigen::MatrixXd& Ubc, int Nghost)
{
    const int N = static_cast<int>(U.rows());
    const int nvars = static_cast<int>(U.cols());
    if (Ubc.rows() != N + 2 * Nghost || Ubc.cols() != nvars)
        throw std::runtime_error("insert_interior_into_ghosted: size mismatch");

    Ubc.block(Nghost, 0, N, nvars) = U;
}

// Extract interior U from ghosted Ubc
inline void extract_interior_from_ghosted(const Eigen::MatrixXd& Ubc, Eigen::MatrixXd& U, int Nghost)
{
    const int N = static_cast<int>(U.rows());
    const int nvars = static_cast<int>(U.cols());
    if (Ubc.rows() != N + 2 * Nghost || Ubc.cols() != nvars)
        throw std::runtime_error("extract_interior_from_ghosted: size mismatch");

    U = Ubc.block(Nghost, 0, N, nvars);
}

// Periodic BC fill on Ubc (ghost rows only)
inline void apply_periodic_bc(Eigen::MatrixXd& Ubc, int N, int Nghost)
{
    const int nvars = static_cast<int>(Ubc.cols());
    if (Ubc.rows() != N + 2 * Nghost)
        throw std::runtime_error("apply_periodic_bc: size mismatch");

    // Left ghosts: rows [0 ...... Nghost-1] copy from last Nghost interior cells
    for (int g = 0; g < Nghost; ++g) {
        const int src_i = N - Nghost + g;
        Ubc.row(g) = Ubc.row(Nghost + src_i);
    }

    // Right ghosts: rows [Nghost+N ..... Nghost+N+Nghost-1] copy from first Nghost interior cells
    for (int g = 0; g < Nghost; ++g) {
        const int dst_row = Nghost + N + g;
        const int src_i   = g;
        Ubc.row(dst_row) = Ubc.row(Nghost + src_i);
    }
}

// Transmissive / outflow (zero-gradient) BC fill on Ubc (ghost rows only).
inline void apply_outflow_bc(Eigen::MatrixXd& Ubc, int N, int Nghost)
{
    const int nvars = static_cast<int>(Ubc.cols());
    if (Ubc.rows() != N + 2 * Nghost)
        throw std::runtime_error("apply_outflow_bc: size mismatch");

    const int first_int = Nghost;
    const int last_int  = Nghost + N - 1;

    // Left ghosts: copy first interior state
    for (int g = 0; g < Nghost; ++g) {
        Ubc.row(g) = Ubc.row(first_int);
    }

    // Right ghosts: copy last interior state
    for (int g = 0; g < Nghost; ++g) {
        Ubc.row(Nghost + N + g) = Ubc.row(last_int);
    }
}

} // namespace fv1d
