#pragma once
#include "Weno1d.H"
#include <Eigen/Dense>
#include <vector>
#include <stdexcept>

namespace weno_adapter {

// Convert Eigen::MatrixXd (rows x cols) -> VectOfVectDouble
inline VectOfVectDouble eigen_to_vov(const Eigen::MatrixXd& A)
{
    VectOfVectDouble out(static_cast<size_t>(A.rows()), VectOfDouble(static_cast<size_t>(A.cols())));
    for (int i = 0; i < A.rows(); ++i)
        for (int j = 0; j < A.cols(); ++j)
            out[static_cast<size_t>(i)][static_cast<size_t>(j)] = A(i, j);
    return out;
}

// Convert VectOfVectDouble (rows x cols) -> Eigen::MatrixXd
inline Eigen::MatrixXd vov_to_eigen(const VectOfVectDouble& V)
{
    if (V.empty()) return Eigen::MatrixXd();
    const int rows = static_cast<int>(V.size());
    const int cols = static_cast<int>(V[0].size());
    Eigen::MatrixXd A(rows, cols);
    for (int i = 0; i < rows; ++i) {
        if (static_cast<int>(V[i].size()) != cols)
            throw std::runtime_error("vov_to_eigen: jagged vector-of-vectors");
        for (int j = 0; j < cols; ++j) A(i, j) = V[i][j];
    }
    return A;
}

// Wrapper: returns UL, UR as (M+1) x nVars, where row d is the d-th spatial derivative scaled by 1/dx^d.
inline void reconstruct_interface(const WENO1d& weno,
                                  const Eigen::MatrixXd& Ubc_eigen,
                                  int i_interface, int Nghost,
                                  Eigen::MatrixXd& UL, Eigen::MatrixXd& UR)
{
    // Convert Ubc into the type WENO wants
    VectOfVectDouble Ubc = eigen_to_vov(Ubc_eigen);

    VectOfVectDouble U_WenoL, U_WenoR;
    weno.WENO_reconstructionForFluxEvaluation(Ubc, i_interface, U_WenoL, U_WenoR, Nghost);

    // WENO outputs (M+1) rows and nVars columns but in vector-of-vectors layout:
    // U_WenoL[d][n] with d=0..M and n=0..nVars-1
    UL = vov_to_eigen(U_WenoL);
    UR = vov_to_eigen(U_WenoR);
}

} // namespace weno_adapter
