#include <stdio.h>
#include <math.h>
#define PI                   acos(-1.)
#define G                    6.67428e-11

double Beta(double e) {
    return pow(1 - e * e, 0.5);
}

double F1(double e) {
    return (1 + 31./2*e*e + 255./8*pow(e,4) + 185./16*pow(e,6) + 25./64*pow(e,8));
}

double F2(double e) {
    return (1 + 15./2*e*e + 45./8*pow(e,4) + 5./16*pow(e,6));
}

double F3(double e) {
    return (1 + 15./4*e*e + 15./8*pow(e,4) + 5./64*pow(e,6));
}

double F4(double e) {
    return (1 + 1.5*e*e + 1./8*pow(e,4));
}

double F5(double e) {
    return (1 + 3*e*e + 3./8*pow(e,4));
}

double Omega(double e, double n) {
    return (1 + 6 * e * e) * n;
}

double dadt(double M, double m, double R, double r, double tau,
            double k2, double a, double e) {
    /*
    Assuming stellar tide is negligible and m << M.
    
    */
    
    double B = Beta(e);
    double n = sqrt(G * M / (a * a * a));
    double w = Omega(e, n);
    double Z = 3 * G * G * k2 * M * M * M * pow(r, 5) * tau / pow(a, 9);
    double sum = Z * (F2(e) * w / (pow(B, 12) * n) - F1(e) / pow(B, 15));
    return 2 * a * a / (G * M * m) * sum;
  
}

double dedt(double M, double m, double R, double r, double tau,
            double k2, double a, double e) {
    /*
    Assuming stellar tide is negligible and m << M.
    
    */

    double B = Beta(e);
    double n = sqrt(G * M / (a * a * a));
    double w = Omega(e, n);
    double Z = 3 * G * G * k2 * M * M * M * pow(r, 5) * tau / pow(a, 9);
    double sum = Z * (F4(e) * w / (pow(B, 10) * n) - (18. / 11.) * F3(e) / pow(B, 13));
    return 11 * a * e / (2 * G * M * m) * sum;
  
}

void Evolve(double M, double m, double R, double r, double tau,
            double k2, double time, double *a, double *e) {
    
    double t = 0;
    double dt, dadt_, dedt_;
    
    while (t <= time) {
        
        // Derivatives
        dadt_ = dadt(M, m, R, r, tau, k2, *a, *e);
        dedt_ = dedt(M, m, R, r, tau, k2, *a, *e);
        
        // Done changing?
        if ((dadt_ == 0) && (dedt_ == 0)) return;

        // Time step
        if ((e == 0) || (fabs(*a / dadt_) < fabs(*e / dedt_)))
            dt = 0.01 * fabs(*a / dadt_);
        else
            dt = 0.01 * fabs(*e / dedt_);
        
        // Force e to zero?
        if (*e < 1e-5) *e = 0;
        
        // Update
        *a += dadt_ * dt;
        *e += dedt_ * dt;
        t += dt;

    }
     
}