#include <cmath>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#define BIGG                    0.00029591220363 // AU^3 / MSUN / DAY^2
#define EPS                     0.01
#define MINE                    1.e-5
#define MEARTH                  3.003e-6 // MSUN
#define RSUN                    4.6491e-3 // AU
#define REARTH                  4.2635e-5
#define SECOND                  (1. / 86400.)
#define YEAR                    365.

using namespace std;
using namespace pybind11::literals;
namespace py = pybind11;

inline double Beta(double& e) {
    return pow(1 - e * e, 0.5);
}

inline double F1(double& e) {
    return (1 + 31./2*e*e + 255./8*pow(e,4) + 185./16*pow(e,6) + 25./64*pow(e,8));
}

inline double F2(double& e) {
    return (1 + 15./2*e*e + 45./8*pow(e,4) + 5./16*pow(e,6));
}

inline double F3(double& e) {
    return (1 + 15./4*e*e + 15./8*pow(e,4) + 5./64*pow(e,6));
}

inline double F4(double& e) {
    return (1 + 1.5*e*e + 1./8*pow(e,4));
}

inline double F5(double& e) {
    return (1 + 3*e*e + 3./8*pow(e,4));
}

inline double Omega(double& e, double n) {
    return (1 + 6 * e * e) * n;
}

inline double dadt(const double& M, const double& m, const double& R,
                   const double& r, const double& tau,
                   const double& k2, double& a, double& e) {
    /*
    Assuming stellar tide is negligible and m << M.

    */

    double B = Beta(e);
    double n = sqrt(BIGG * M / (a * a * a));
    double w = Omega(e, n);
    double Z = 3 * BIGG * BIGG * k2 * M * M * M * pow(r, 5) * tau / pow(a, 9);
    double sum = Z * (F2(e) * w / (pow(B, 12) * n) - F1(e) / pow(B, 15));
    return 2 * a * a / (BIGG * M * m) * sum;

}

inline double dedt(const double& M, const double& m, const double& R, const double& r,
                   const double& tau, const double& k2, double& a, double& e) {
    /*
    Assuming stellar tide is negligible and m << M.

    */

    double B = Beta(e);
    double n = sqrt(BIGG * M / (a * a * a));
    double w = Omega(e, n);
    double Z = 3 * BIGG * BIGG * k2 * M * M * M * pow(r, 5) * tau / pow(a, 9);
    double sum = Z * (F4(e) * w / (pow(B, 10) * n) - (18. / 11.) * F3(e) / pow(B, 13));
    return 11 * a * e / (2 * BIGG * M * m) * sum;

}

vector<double> evolve(double M, double m, double R, double r, double tau, double k2,
                      double time, double a, double e) {

    // Convert to AU, MSUN, and DAY
    m *= MEARTH;
    R *= RSUN;
    r *= REARTH;
    tau *= SECOND;
    time *= YEAR;
    double t = 0;
    double dt, dadt_, dedt_;
    vector<double> ae;

    while (t <= time) {

        // Derivatives
        dadt_ = dadt(M, m, R, r, tau, k2, a, e);
        dedt_ = dedt(M, m, R, r, tau, k2, a, e);

        // Done changing?
        if ((dadt_ == 0) && (dedt_ == 0)) break;

        // Time step
        if ((e == 0) || (abs(a / dadt_) < abs(e / dedt_)))
            dt = EPS * abs(a / dadt_);
        else
            dt = EPS * abs(e / dedt_);

        // Update
        a += dadt_ * dt;
        e += dedt_ * dt;
        t += dt;

        // Force e to zero?
        if (e < MINE) e = 0;

    }

    ae.push_back(a);
    ae.push_back(e);
    return ae;

}

PYBIND11_MODULE(tidal, m) {
    m.doc() = R"pbdoc(
        Tidal evolution equation solver (CTL).
    )pbdoc";

    m.def("evolve", &evolve, R"pbdoc(
        Evolve a planet forward in time.
    )pbdoc", "M"_a=1., "m"_a=1., "R"_a=1., "r"_a=1., "tau"_a=100.,
    "k2"_a=1., "time"_a=1.e9, "a"_a=0.01, "e"_a=0.3);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

}
