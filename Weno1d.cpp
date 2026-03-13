/* ------------------------------------------------------------------ */
/*                                                                    */
/*                       WENO RECONSTRUCTION                          */
/*                                                                    */
/* ------------------------------------------------------------------ */
#include "Weno1d.H"

WENO1d::WENO1d(int Polydegree, double dx)
{
    /* Polydegree is the degree of the polynomials used for the reconstruction (1-> linear slopes, 2-> parabolas); dx is the mesh spacing*/

  	m_dx = dx;
  	m_M = Polydegree;

    m_TotalStencilsize = 2*Polydegree+1;
    m_PolyStencilsize = Polydegree+1;

    if(m_M > 2)
  	{
  		std::cerr << "The WENO reconstruction is implemented for polynomials with degree up to 2 (i.e to construct a third order ADER method). Program stopped.";
  		std::exit(EXIT_FAILURE);
  	}

    m_CMatsL.resize(m_M+1);
    m_CMatsR.resize(m_M+1);
    m_BMats.resize(m_M+1);

    for(int i=0; i<m_M+1; i++)
    {
        m_CMatsL[i].resize(m_M+1,m_M+1);
        m_CMatsR[i].resize(m_M+1,m_M+1);
        m_BMats[i].resize(m_M+1,m_M+1);
    }

    m_optWeightsL.resize(m_M+1);
    m_optWeightsR.resize(m_M+1);

    setCoefficients();
}

void WENO1d::WENO_reconstructionForFluxEvaluation(const VectOfVectDouble& Ubc, const int i_interface, VectOfVectDouble& U_WenoL, VectOfVectDouble& U_WenoR, const int Nghost) const
{
    /* Ubc is a matrix (N+2*Nghost) x NVARS
       i_interface is the index of the interface at which reconstruct to the left and to the right using WENO
       U_WenoL contains the reconstructed poly to the left  of the interface.     [0,1] is the zero-th derivative for the variable 1, [1,0] is the first derivative for the variable 0
       U_WenoR contains the reconstructed poly to the right of the interface. 
       Nghost is the number of ghost cells used in Ubc matrix    
    */

    int nVars = Ubc[0].size();
    int numberofGhostRequired = m_M+1;
    
    if(Nghost < numberofGhostRequired)
    {
        std::cerr << "The WENO reconstruction requires at least " << numberofGhostRequired << " ghost cells. Change the number of ghost cells used in your UBC matrix.";
        std::exit(EXIT_FAILURE);
    }

    if(U_WenoL.size() != m_M+1 || U_WenoL[0].size() != nVars)
    {
        U_WenoL.resize(m_M+1,VectOfDouble(nVars));
    }
    if(U_WenoR.size() != m_M+1 || U_WenoR[0].size() != nVars)
    {
        U_WenoR.resize(m_M+1,VectOfDouble(nVars));
    }

    if(m_M == 0)
    {
        U_WenoL[0] = Ubc[i_interface+Nghost-1];
    	U_WenoR[0] = Ubc[i_interface+Nghost];

        return;
    }

    Eigen::VectorXd U1l(2*m_M+1);
    Eigen::VectorXd U1r(2*m_M+1);
    Eigen::VectorXd dqWENOL(m_M+1);
    Eigen::VectorXd dqWENOR(m_M+1);

    Eigen::MatrixXd stencil(2*m_M+2,nVars);

    getStencil(Ubc,i_interface,stencil,Nghost);

    for(int n=0; n<nVars; n++)
    {
        for(int j=0; j<2*m_M+1; j++)
      	{
      		U1l[j] = stencil(j,n);
            U1r[j] = stencil(j+1,n);
      	}

        reconstructAtInterface(U1l,U1r,dqWENOL,dqWENOR);

        for(int d=0; d<m_M+1; d++)
        {
            U_WenoL[d][n] = dqWENOL(d);
            U_WenoR[d][n] = dqWENOR(d);
        }
    }
}

void WENO1d::getStencil(const VectOfVectDouble& Ubc, const int i_interface, Eigen::MatrixXd& stencil, const int Nghost) const
{
    int nVars = Ubc[0].size();

    int numberofGhostRequired = m_M+1;

    for(int j=0; j<2*m_M+2; j++)
    {
        for(int n=0; n<nVars; n++)
        {
            stencil(j,n) = Ubc[i_interface+Nghost-numberofGhostRequired+j][n];
        }
    }
}

void WENO1d::reconstructAtInterface(const Eigen::VectorXd& U1l, const Eigen::VectorXd& U1r, Eigen::VectorXd& dqWENOL, Eigen::VectorXd& dqWENOR) const
{
    if(dqWENOL.size() != m_M+1 || dqWENOR.size() != m_M+1 ){dqWENOL.resize(m_M+1); dqWENOR.resize(m_M+1);}

    Eigen::VectorXd WeightsL(m_M+1),WeightsR(m_M+1);

    computeWeights(U1l,U1r,WeightsL,WeightsR);

    dqWENOL.array() = 0.0;
    dqWENOR.array() = 0.0;

    for(int i=0; i<m_M+1; i++)
    {
        Eigen::VectorXd dPL = WeightsL(i)*(m_CMatsL[i]*(U1l.segment(i,m_M+1)));
        Eigen::VectorXd dPR = WeightsR(i)*(m_CMatsR[i]*(U1r.segment(i,m_M+1)));

        dqWENOL += dPL;
        dqWENOR += dPR;
    }

    for(int j=0; j<m_M+1; j++)
    {
        dqWENOL(j) *= 1.0/pow(m_dx,j);
        dqWENOR(j) *= 1.0/pow(m_dx,j);
    }
}

void WENO1d::computeWeights(const Eigen::VectorXd& U1l, const Eigen::VectorXd& U1r, Eigen::VectorXd& WeightsL, Eigen::VectorXd& WeightsR) const
{
    // compute smoothness Indicators

    Eigen::VectorXd SmoothnessIndL(m_M+1);
    Eigen::VectorXd SmoothnessIndR(m_M+1);

    for(int i=0; i<m_M+1; i++)
    {
        SmoothnessIndL[i] = (U1l.segment(i,m_M+1)).transpose()*(m_BMats[i]*U1l.segment(i,m_M+1));
        SmoothnessIndR[i] = (U1r.segment(i,m_M+1)).transpose()*(m_BMats[i]*U1r.segment(i,m_M+1));
    }

    double SumWeightsL = 0.0;
    double SumWeightsR = 0.0;

    for(int i=0; i<m_M+1; i++)
    {
        WeightsL[i] = m_optWeightsL[i]/pow(m_epsilon+SmoothnessIndL[i],m_qPower);
        WeightsR[i] = m_optWeightsR[i]/pow(m_epsilon+SmoothnessIndR[i],m_qPower);

        SumWeightsL += WeightsL[i];
        SumWeightsR += WeightsR[i];
    }

    WeightsL /= SumWeightsL;
    WeightsR /= SumWeightsR;
}

void WENO1d::setCoefficients(void)
{
    switch(m_M)
    {
        case 1:
        {
            m_CMatsL[0] << -0.5, 1.5, -1.0, 1.0;
            m_CMatsL[1] <<  0.5, 0.5, -1.0, 1.0;

            m_CMatsR[0] << 0.5, 0.5, -1, 1;
            m_CMatsR[1] << 1.5, -0.5, -1, 1;

            m_BMats[0] << 1, -1, -1, 1;
            m_BMats[1] << 1, -1, -1, 1;

            m_optWeightsL << 0.3333333333333333, 0.6666666666666667;
            m_optWeightsR << 0.6666666666666667, 0.3333333333333333;

            break;
        }
        case 2:
        {
            m_CMatsL[0] << 0.3333333333333333, -1.166666666666667, 1.833333333333333, 1, -3, 2, 1, -2, 1;
            m_CMatsL[1] << -0.1666666666666667, 0.8333333333333334, 0.3333333333333333, 0, -1, 1, 1, -2, 1;
            m_CMatsL[2] << 0.3333333333333333, 0.8333333333333334, -0.1666666666666667, -1, 1, 0, 1, -2, 1;

            m_CMatsR[0] << -0.1666666666666667, 0.8333333333333334, 0.3333333333333333, 0, -1, 1, 1, -2, 1;
            m_CMatsR[1] << 0.3333333333333333, 0.8333333333333334, -0.1666666666666667, -1, 1, 0, 1, -2, 1;
            m_CMatsR[2] << 1.833333333333333, -1.166666666666667, 0.3333333333333333, -2, 3, -1, 1, -2, 1;

            m_BMats[0] << 1.333333333333333, -3.166666666666667, 1.833333333333333, -3.166666666666667, 8.333333333333334, -5.166666666666667, 1.833333333333333, -5.166666666666667, 3.333333333333333;
            m_BMats[1] << 1.333333333333333, -2.166666666666667, 0.8333333333333334, -2.166666666666667, 4.333333333333333, -2.166666666666667, 0.8333333333333334, -2.166666666666667, 1.333333333333333;
            m_BMats[2] << 3.333333333333333, -5.166666666666667, 1.833333333333333, -5.166666666666667, 8.333333333333334, -3.166666666666667, 1.833333333333333, -3.166666666666667, 1.333333333333333;

            m_optWeightsL << 0.1, 0.6, 0.3;
            m_optWeightsR << 0.3, 0.6, 0.1;

            break;
        }
        default:
        {
            std::cerr << "degree not implemented" << std::endl;
            std::exit(EXIT_FAILURE);
            break;
        }
    }
}