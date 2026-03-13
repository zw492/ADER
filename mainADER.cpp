#include "Euler1D.h"
#include "Grid1D.h"
#include "TimeStep.h"
#include "MusclHancock1D.h"
#include "WenoAdapter.h"
#include "Weno1d.H"
#include "CK_Euler.h"
#include "ADER3_Flux.h"
#include "Step8_Verification.h"
#include "ExactRiemannEuler1D.h"
#include "Step22_RiemannTests.h"
#include "Step23_ShockTurbulence.h"
#include "Step24_Efficiency.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace euler1d;

static void print_rows(const Eigen::MatrixXd& A, int r0, int r1, const std::string& label)
{
    std::cout << label << "\n";
    for (int r = r0; r <= r1; ++r) {
        std::cout << "row " << r << " : " << A.row(r) << "\n";
    }
    std::cout << "\n";
}

int main()
{   
    std::cout << "=== Euler1D Core Physics Test ===\n\n";
    

    // Step 1: Euler core physics sanity test

    Params par;
    par.gamma = 1.4;
    par.strict_checks = true;

    Vec3 W;
    W << 1.0, 2.0, 1.0;

    Vec3 Q = primToCons(W, par);
    Vec3 W_back = consToPrim(Q, par);

    std::cout << "Primitive W      = " << W.transpose() << "\n";
    std::cout << "Conserved Q      = " << Q.transpose() << "\n";
    std::cout << "Recovered W_back = " << W_back.transpose() << "\n\n";

    std::cout << "Round-trip error = " << (W - W_back).norm() << "\n\n";

    Vec3 F = flux(Q, par);
    std::cout << "Flux F(Q) = " << F.transpose() << "\n\n";

    double c = soundSpeedFromPrim(consToPrim(Q, par), par);
    double amax = maxWaveSpeed(Q, par);
    std::cout << "Sound speed c    = " << c << "\n";
    std::cout << "Max wave speed   = |u| + c = " << amax << "\n\n";

    // CFL dt quick test
    {
        int N = 5;
        double dx = 0.1, CFL = 0.9;
        std::vector<Vec3> U(N);

        for (int i = 0; i < N; ++i) {
            Vec3 Wi; Wi << 1.0 + 0.1*i, 2.0, 1.0;
            U[i] = primToCons(Wi, par);
        }

        double amax_grid = 0.0;
        for (int i = 0; i < N; ++i) amax_grid = std::max(amax_grid, maxWaveSpeed(U[i], par));
        double dt = CFL * dx / amax_grid;

        std::cout << "CFL timestep test on small grid:\n";
        std::cout << "Computed dt = " << dt << "\n\n";
    }

    std::cout << "=== Step 1 completed successfully ===\n\n";


    // Step 2: Grid + layout + ghost cells + BC fill

    std::cout << "=== Step 2: Grid + ghost cells + BCs ===\n\n";

    const double xmin = 0.0, xmax = 1.0;
    const int N = 100;
    const int Nghost = 3;

    fv1d::Grid1D grid(xmin, xmax, N);
    std::cout << "Grid: xmin=" << xmin << ", xmax=" << xmax << ", N=" << N << ", dx=" << grid.dx << "\n\n";

    // Allocate U (interior) and Ubc (ghosted)
    Eigen::MatrixXd U, Ubc;
    fv1d::allocate_state(N, Nghost, U, Ubc, /*nvars=*/3);

    // Fill interior with a smooth periodic state (cell-center sample)
    const double pi = 3.14159265358979323846;
    for (int i = 0; i < N; ++i) {
        double x = grid.xc[i];
        double rho = 1.0 + 0.2 * std::sin(2.0 * pi * x);
        double u   = 1.0;
        double p   = 1.0;

        Vec3 Wi; Wi << rho, u, p;
        Vec3 Qi = primToCons(Wi, par);
        U.row(i) = Qi.transpose();
    }

    fv1d::insert_interior_into_ghosted(U, Ubc, Nghost);
    fv1d::apply_periodic_bc(Ubc, N, Nghost);

    // Validate: left ghost rows should match last interior rows, right ghosts match first interior rows
    print_rows(Ubc, 0, Nghost - 1, "Periodic BC: Left ghost rows (should equal last interior cells)");
    print_rows(Ubc, Nghost, Nghost + 2, "Periodic BC: First 3 interior rows");
    print_rows(Ubc, Nghost + N - 3, Nghost + N - 1, "Periodic BC: Last 3 interior rows");
    print_rows(Ubc, Nghost + N, Nghost + N + Nghost - 1, "Periodic BC: Right ghost rows (should equal first interior cells)");

    fv1d::insert_interior_into_ghosted(U, Ubc, Nghost);
    fv1d::apply_outflow_bc(Ubc, N, Nghost);

    print_rows(Ubc, 0, Nghost - 1, "Outflow BC: Left ghost rows (should equal first interior cell)");
    print_rows(Ubc, Nghost, Nghost, "Outflow BC: First interior row");
    print_rows(Ubc, Nghost + N - 1, Nghost + N - 1, "Outflow BC: Last interior row");
    print_rows(Ubc, Nghost + N, Nghost + N + Nghost - 1, "Outflow BC: Right ghost rows (should equal last interior cell)");

    std::cout << "=== Step 2 completed successfully ===\n";

    // Step 3: Time step controller (CFL)
    std::cout << "\n=== Step 3: CFL time-step controller ===\n\n";

    const double CFL = 0.9;
    const double dt_cfl = compute_dt_cfl(U, grid.dx, CFL, par);

    std::cout << "CFL = " << CFL << "\n";
    std::cout << "dx  = " << grid.dx << "\n";
    std::cout << "Computed dt (from interior U) = " << dt_cfl << "\n";

    Eigen::MatrixXd U_from_Ubc = Ubc.block(Nghost, 0, N, 3);
    const double dt_cfl2 = compute_dt_cfl(U_from_Ubc, grid.dx, CFL, par);
    std::cout << "Computed dt (from Ubc interior) = " << dt_cfl2 << "\n";
    std::cout << "Difference |dt1-dt2| = " << std::abs(dt_cfl - dt_cfl2) << "\n\n";

    std::cout << "=== Step 3 completed successfully ===\n";


    // Step 4: MUSCL–Hancock baseline (2nd order TVD)

    std::cout << "\n=== Step 4: MUSCL–Hancock baseline ===\n\n";

    // Use periodic BC for this smooth advection-style validation
    // (rho varies smoothly, u=1, p=1 -> contact-like advection; should stay stable)
    const double Tfinal = 0.2;
    double t = 0.0;

    Eigen::MatrixXd U_n = U;
    Eigen::MatrixXd Ubc_n = Ubc;
    Eigen::MatrixXd U_np1;

    int stepCount = 0;
    while (t < Tfinal) {
        // Fill ghosted array and apply periodic BC
        fv1d::insert_interior_into_ghosted(U_n, Ubc_n, Nghost);
        fv1d::apply_periodic_bc(Ubc_n, N, Nghost);

        // CFL dt
        double dt = compute_dt_cfl(U_n, grid.dx, /*CFL=*/0.5, par);
        if (t + dt > Tfinal) dt = Tfinal - t; // For ALL actuall tasks we use CFL=0.9 throughout

        muscl1d::step(Ubc_n, N, Nghost, dt, grid.dx, par, U_np1);
        U_n.swap(U_np1);
        t += dt;
        stepCount++;

        // Simple diagnostics
        double rho_min = 1e300, p_min = 1e300;
        for (int i = 0; i < N; ++i) {
            euler1d::Vec3 Qi = U_n.row(i).transpose();
            euler1d::Vec3 Wi = euler1d::consToPrim(Qi, par);
            rho_min = std::min(rho_min, Wi(0));
            p_min   = std::min(p_min,   Wi(2));
        }

        if (stepCount % 10 == 0 || t >= Tfinal) {
            std::cout << "t=" << t << ", step=" << stepCount
                      << ", rho_min=" << rho_min << ", p_min=" << p_min << "\n";
        }
    }

    std::cout << "\nSample cell comparison (rho only):\n";
    for (int i : {0, N/2, N-1}) {
        double rho0 = euler1d::consToPrim(U.row(i).transpose(), par)(0);
        double rhoT = euler1d::consToPrim(U_n.row(i).transpose(), par)(0);
        std::cout << "cell " << i << ": rho(t=0)=" << rho0 << ", rho(t=" << Tfinal << ")=" << rhoT << "\n";
    }

    std::cout << "\n=== Step 4 completed (MUSCL–Hancock runs stably) ===\n";


    // Step 5: Integrate provided WENO reconstruction (for ADER)

    std::cout << "\n=== Step 5: WENO reconstruction wiring test ===\n\n";
    const int PolyDegree = 2;
    WENO1d weno(PolyDegree, grid.dx);

    fv1d::insert_interior_into_ghosted(U, Ubc, Nghost);
    fv1d::apply_periodic_bc(Ubc, N, Nghost);

    auto rho_exact = [&](double x) {
        return 1.0 + 0.2 * std::sin(2.0 * pi * x);
    };
    auto drho_dx_exact = [&](double x) {
        return 0.4 * pi * std::cos(2.0 * pi * x);
    };
    auto d2rho_dx2_exact = [&](double x) {
        return -0.8 * pi * pi * std::sin(2.0 * pi * x);
    };

    for (int j : {0, 1, N/2, N-1, N}) {
        const double x_if = xmin + j * grid.dx;

        Eigen::MatrixXd UL, UR; // (M+1) x 3
        weno_adapter::reconstruct_interface(weno, Ubc, j, Nghost, UL, UR);

        // For variable 0 (rho), WENO returns:
        // UL(0,0) ~ rho_L at interface, UL(1,0) ~ (d rho/dx)_L, UL(2,0) ~ (d2 rho/dx2)_L
        // Similar for UR
        const double rhoL = UL(0,0), rhoR = UR(0,0);
        const double rho_x_L  = UL(1,0), rho_x_R  = UR(1,0);
        const double rho_xx_L = UL(2,0), rho_xx_R = UR(2,0);

        const double rhoE = rho_exact(x_if);
        const double rho_x_E  = drho_dx_exact(x_if);
        const double rho_xx_E = d2rho_dx2_exact(x_if);

        std::cout << "Interface j=" << j << " at x=" << x_if << "\n";
        std::cout << "  rho exact      = " << rhoE << "\n";
        std::cout << "  rhoL, rhoR     = " << rhoL << " , " << rhoR << "\n";
        std::cout << "  drho/dx exact  = " << rho_x_E << "\n";
        std::cout << "  rho_x L,R      = " << rho_x_L << " , " << rho_x_R << "\n";
        std::cout << "  d2rho/dx2 exact= " << rho_xx_E << "\n";
        std::cout << "  rho_xx L,R     = " << rho_xx_L << " , " << rho_xx_R << "\n\n";
    }

    std::cout << "=== Step 5 completed (WENO called and produced interface values/derivatives) ===\n";

    // Step 6: CK procedure (compute Qt, Qtt from Q, Qx, Qxx)

    std::cout << "\n=== Step 6: CK (Qt, Qtt) validation ===\n\n";
    // Analytic smooth field: rho(x)=1+0.2 sin(2pi x), u=1, p=1.
    // Build Q, Qx, Qxx analytically at a point and compare CK result to known exact.
    const double PI = 3.14159265358979323846;
    const double x0 = 0.37;

    const double rho0 = 1.0 + 0.2 * std::sin(2.0 * PI * x0);
    const double rho_x  = 0.4 * PI * std::cos(2.0 * PI * x0);
    const double rho_xx = -0.8 * PI * PI * std::sin(2.0 * PI * x0);

    const double u0 = 1.0;
    const double p0 = 1.0;

    euler1d::Vec3 W0; W0 << rho0, u0, p0;
    euler1d::Vec3 Q0 = euler1d::primToCons(W0, par);

    // Derivatives of conserved variables:
    // m = rho*u, with u const => m_x = rho_x, m_xx=rho_xx
    // E = p/(g-1) + 0.5*rho*u^2, with p,u const => E_x=0.5*u^2*rho_x, E_xx=0.5*u^2*rho_xx
    euler1d::Vec3 Qx0, Qxx0;
    Qx0  << rho_x,  rho_x,  0.5 * rho_x;
    Qxx0 << rho_xx, rho_xx, 0.5 * rho_xx;

    euler1d::Vec3 Qt0, Qtt0;
    CK_Euler_qt_qtt(Q0, Qx0, Qxx0, par, Qt0, Qtt0);

    // Exact for this special case:
    euler1d::Vec3 Qt_exact, Qtt_exact;
    Qt_exact  << -rho_x,  -rho_x,  -0.5 * rho_x;
    Qtt_exact <<  rho_xx,  rho_xx,  0.5 * rho_xx;

    std::cout << "x0 = " << x0 << "\n";
    std::cout << "Q0      = " << Q0.transpose() << "\n";
    std::cout << "Qx0     = " << Qx0.transpose() << "\n";
    std::cout << "Qxx0    = " << Qxx0.transpose() << "\n\n";

    std::cout << "Qt (CK)     = " << Qt0.transpose() << "\n";
    std::cout << "Qt (exact)  = " << Qt_exact.transpose() << "\n";
    std::cout << "||Qt-Qt_e|| = " << (Qt0 - Qt_exact).norm() << "\n\n";

    std::cout << "Qtt (CK)     = " << Qtt0.transpose() << "\n";
    std::cout << "Qtt (exact)  = " << Qtt_exact.transpose() << "\n";
    std::cout << "||Qtt-Qtt_e|| = " << (Qtt0 - Qtt_exact).norm() << "\n\n";

    std::cout << "=== Step 6 completed (CK Qt/Qtt computed) ===\n";


    // Step 7: ADER3 interface flux + one FV update test
    std::cout << "\n=== Step 7: ADER3 interface flux test ===\n\n";

    {
        // Unique names to avoid clashes
        const int PolyDegree_ADER = 2;
        WENO1d weno_ADER(PolyDegree_ADER, grid.dx);

        Eigen::MatrixXd U_ader = U;       
        Eigen::MatrixXd Ubc_ader;         
        Eigen::MatrixXd Unp1_ader;        

        const double dt_ader = compute_dt_cfl(U_ader, grid.dx, 0.4, par);
        // For ALL actuall tasks we use CFL=0.9 throughout

        // Build ghosted array and apply periodic BC (for this smooth test)
        Ubc_ader.resize(N + 2 * Nghost, 3);
        fv1d::insert_interior_into_ghosted(U_ader, Ubc_ader, Nghost);
        fv1d::apply_periodic_bc(Ubc_ader, N, Nghost);

        std::vector<euler1d::Vec3> Fbar(N + 1);
        for (int j = 0; j <= N; ++j) {
            Fbar[j] = ADER3_interface_flux(weno_ADER, Ubc_ader, j, Nghost, dt_ader, par);
        }

        // FV update using current state U_ader
        Unp1_ader.resize(N, 3);
        for (int i = 0; i < N; ++i) {
            const euler1d::Vec3 Qi = U_ader.row(i).transpose();
            const euler1d::Vec3 Qnew = Qi - (dt_ader / grid.dx) * (Fbar[i + 1] - Fbar[i]);
            Unp1_ader.row(i) = Qnew.transpose();
        }

        // Diagnostics
        double rho_min = 1e300, p_min = 1e300;
        for (int i = 0; i < N; ++i) {
            const euler1d::Vec3 Qi = Unp1_ader.row(i).transpose();
            const euler1d::Vec3 Wi = euler1d::consToPrim(Qi, par);
            rho_min = std::min(rho_min, Wi(0));
            p_min   = std::min(p_min,   Wi(2));
        }

        std::cout << "dt_ader = " << dt_ader << "\n";
        std::cout << "After one ADER3 step: rho_min=" << rho_min << ", p_min=" << p_min << "\n";
    }

    
    // Step 8: Verification infrastructure (cell averages, norms, timing, convergence)
    std::cout << "\n=== Step 8: Verification + experiments infrastructure ===\n\n";
{
    // domain [-1,1], periodic; rho=2+sin^4(pi x), u=1, p=1; Tout=2
    const int Ng = std::max(Nghost, 3);
    std::vector<int> Ns = {25, 50, 100, 200, 400};

    const double Tfinal = 2.0;
    const double CFL_MUSCL = 0.9;
    const double CFL_ADER  = 0.9;
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "PDF convergence test at T=" << Tfinal << "\n\n";

    const std::string csv_name = "convergence_table_T2.csv";
    std::ofstream csv(csv_name);
    if (!csv) {
        std::cerr << "ERROR: could not open " << csv_name << " for writing.\n";
        return 1;
    }

    csv << std::scientific << std::setprecision(6);
    csv << "scheme,N,dx,L1,L2,Linf,rate_L1,rate_L2,rate_Linf,runtime_sec\n";

    std::cout << "CSV columns:\n";
    std::cout << "scheme,N,dx,L1,L2,Linf,rate_L1,rate_L2,rate_Linf,runtime_sec\n\n";

    auto rate = [](double e_coarse, double e_fine, double dx_coarse, double dx_fine) {
        return std::log(e_coarse / e_fine) / std::log(dx_coarse / dx_fine);
    };

    bool muscl_has_prev = false;
    double muscl_prev_dx   = 0.0;
    double muscl_prev_L1   = 0.0;
    double muscl_prev_L2   = 0.0;
    double muscl_prev_Linf = 0.0;

    bool ader_has_prev = false;
    double ader_prev_dx   = 0.0;
    double ader_prev_L1   = 0.0;
    double ader_prev_L2   = 0.0;
    double ader_prev_Linf = 0.0;

    for (int Ntest : Ns) {

        fv1d::Grid1D g(-1.0, 1.0, Ntest);

        Eigen::MatrixXd U0 = step8::init_pdf_cell_averages(g, par);

        std::vector<double> rho_exact = step8::exact_rho_cell_averages_pdf(g, Tfinal);

        // MUSCL
        Eigen::MatrixXd U_muscl;
        double t_muscl = step8::time_seconds([&](){
            U_muscl = step8::evolve_MUSCL(U0, g, Ng, par, CFL_MUSCL, Tfinal);
        });
        step8::Norms nm = step8::rho_error_norms(U_muscl, g, rho_exact);

        double muscl_rate_L1   = std::numeric_limits<double>::quiet_NaN();
        double muscl_rate_L2   = std::numeric_limits<double>::quiet_NaN();
        double muscl_rate_Linf = std::numeric_limits<double>::quiet_NaN();

        if (muscl_has_prev) {
            muscl_rate_L1   = rate(muscl_prev_L1,   nm.L1,   muscl_prev_dx, g.dx);
            muscl_rate_L2   = rate(muscl_prev_L2,   nm.L2,   muscl_prev_dx, g.dx);
            muscl_rate_Linf = rate(muscl_prev_Linf, nm.Linf, muscl_prev_dx, g.dx);
        }

        std::cout << "MUSCL," << Ntest << "," << g.dx << ","
                  << nm.L1 << "," << nm.L2 << "," << nm.Linf << ","
                  << muscl_rate_L1 << "," << muscl_rate_L2 << "," << muscl_rate_Linf << ","
                  << t_muscl << "\n";

        csv << "MUSCL," << Ntest << "," << g.dx << ","
            << nm.L1 << "," << nm.L2 << "," << nm.Linf << ","
            << muscl_rate_L1 << "," << muscl_rate_L2 << "," << muscl_rate_Linf << ","
            << t_muscl << "\n";

        muscl_has_prev  = true;
        muscl_prev_dx   = g.dx;
        muscl_prev_L1   = nm.L1;
        muscl_prev_L2   = nm.L2;
        muscl_prev_Linf = nm.Linf;

        if (Ntest == 100) {
            std::cout << "\n--- DIAGNOSTIC at T=" << Tfinal << " for N=100 (MUSCL) ---\n";

            std::vector<int> samp = {0, Ntest/4, Ntest/2, 3*Ntest/4, Ntest-1};
            for (int i : samp) {
                double x = g.xc[i];
                double rho_num = U_muscl(i, 0);
                double rho_ex  = rho_exact[i];
                std::cout << "i=" << i << " x=" << x
                          << " rho_num=" << rho_num
                          << " rho_ex="  << rho_ex
                          << " diff="    << (rho_num - rho_ex) << "\n";
            }

            double rho_min = 1e300, rho_max = -1e300;
            double u_min   = 1e300, u_max   = -1e300;
            double p_min   = 1e300, p_max   = -1e300;

            for (int i = 0; i < Ntest; ++i) {
                euler1d::Vec3 Q = U_muscl.row(i).transpose();
                euler1d::Vec3 W = euler1d::consToPrim(Q, par); // [rho,u,p]
                rho_min = std::min(rho_min, W(0)); rho_max = std::max(rho_max, W(0));
                u_min   = std::min(u_min,   W(1)); u_max   = std::max(u_max,   W(1));
                p_min   = std::min(p_min,   W(2)); p_max   = std::max(p_max,   W(2));
            }

            std::cout << "rho in [" << rho_min << ", " << rho_max << "]\n";
            std::cout << "u   in [" << u_min   << ", " << u_max   << "]\n";
            std::cout << "p   in [" << p_min   << ", " << p_max   << "]\n";
            std::cout << "---------\n\n";
        }

        // ADER3
        Eigen::MatrixXd U_ader;
        double t_ader = step8::time_seconds([&](){
            U_ader = step8::evolve_ADER3(U0, g, Ng, par, CFL_ADER, Tfinal);
        });
        step8::Norms na = step8::rho_error_norms(U_ader, g, rho_exact);

        double ader_rate_L1   = std::numeric_limits<double>::quiet_NaN();
        double ader_rate_L2   = std::numeric_limits<double>::quiet_NaN();
        double ader_rate_Linf = std::numeric_limits<double>::quiet_NaN();

        if (ader_has_prev) {
            ader_rate_L1   = rate(ader_prev_L1,   na.L1,   ader_prev_dx, g.dx);
            ader_rate_L2   = rate(ader_prev_L2,   na.L2,   ader_prev_dx, g.dx);
            ader_rate_Linf = rate(ader_prev_Linf, na.Linf, ader_prev_dx, g.dx);
        }

        std::cout << "ADER3," << Ntest << "," << g.dx << ","
                  << na.L1 << "," << na.L2 << "," << na.Linf << ","
                  << ader_rate_L1 << "," << ader_rate_L2 << "," << ader_rate_Linf << ","
                  << t_ader << "\n";

        csv << "ADER3," << Ntest << "," << g.dx << ","
            << na.L1 << "," << na.L2 << "," << na.Linf << ","
            << ader_rate_L1 << "," << ader_rate_L2 << "," << ader_rate_Linf << ","
            << t_ader << "\n";

        ader_has_prev  = true;
        ader_prev_dx   = g.dx;
        ader_prev_L1   = na.L1;
        ader_prev_L2   = na.L2;
        ader_prev_Linf = na.Linf;
    }

    csv.close();

    std::cout << "\nCSV written to: " << csv_name << "\n";
    std::cout << "Note: Use these CSV lines to make log-log plots (error vs dx, time vs error).\n";
}

std::cout << "\n=== Step 8 completed (PDF convergence test printed) ===\n";


// 2.2 Riemann tests
{
    Params par;
    par.gamma = 1.4;
    par.strict_checks = true;

    const double CFL = 0.9;

    // Sod: rhoL=1, uL=0, pL=1 ; rhoR=0.125, uR=0, pR=0.1 ; x0=0.5 ; Tout=0.2
    Vec3 WL_sod; WL_sod << 1.0,   0.0, 1.0;
    Vec3 WR_sod; WR_sod << 0.125, 0.0, 0.1;
    step22::run_test("Sod", WL_sod, WR_sod, /*x0=*/0.5, /*Tout=*/0.2,
                     /*N=*/200, CFL, par, /*also_run_muscl=*/true); 

    // Lax: rhoL=0.445,uL=0.698,pL=3.528 ; rhoR=0.5,uR=0,pR=0.571 ; x0=0.6 ; Tout=0.14
    Vec3 WL_lax; WL_lax << 0.445, 0.698, 3.528;
    Vec3 WR_lax; WR_lax << 0.5,   0.0,   0.571;
    step22::run_test("Lax", WL_lax, WR_lax, /*x0=*/0.6, /*Tout=*/0.14,
                     /*N=*/200, CFL, par, /*also_run_muscl=*/true); 
}

step23::run();
step24::run();

return 0;
}