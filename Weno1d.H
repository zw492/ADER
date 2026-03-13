#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <math.h>
#include <algorithm>
#include <vector>
#include <functional>
#include <chrono>

#include <Eigen/Core>
#include <Eigen/Dense>

typedef std::vector<double> VectOfDouble;
typedef std::vector<std::vector<double>> VectOfVectDouble;
typedef std::vector<std::vector<std::vector<double>>> Vect3Double;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;

// WENO CLASS

#ifndef _WENO_h_
#define _WENO_h_
class WENO1d
{
public:

	WENO1d(int PolyDegree, double dx);

	int m_M;
  int m_TotalStencilsize;
  int m_PolyStencilsize;
	double m_dx;

  void WENO_reconstructionForFluxEvaluation(const VectOfVectDouble& Ubc, const int i_interface, VectOfVectDouble& U_WenoL, VectOfVectDouble& U_WenoR, const int Nghost) const;

protected:

  std::vector<MatrixXd> m_CMatsL;
  std::vector<MatrixXd> m_CMatsR;

  std::vector<MatrixXd> m_BMats;

  Eigen::VectorXd m_optWeightsL;
  Eigen::VectorXd m_optWeightsR;

  void getStencil(const VectOfVectDouble& Ubc, const int i_interface, Eigen::MatrixXd& stencil, const int Nghost) const;

  void reconstructAtInterface(const Eigen::VectorXd& U1l, const Eigen::VectorXd& U1r, Eigen::VectorXd& dqWENOL, Eigen::VectorXd& dqWENOR) const;

  void computeWeights(const Eigen::VectorXd& U1l, const Eigen::VectorXd& U1r, Eigen::VectorXd& WeightsL, Eigen::VectorXd& WeightsR) const;

private:

  double m_epsilon = 1.0E-6;
  int m_qPower = 2;

  void setCoefficients(void);

};
#endif
