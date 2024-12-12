#include <cmath>
#include <iostream>


using namespace std;



void conjugateGradient(double **A, double *b, double *x, int n, double tol = 1e-6, int maxIter = 1000) {
    double *r = new double[n];
    double *p = new double[n];
    double *Ap = new double[n];

    // Initial residual r_0 = b - A*x_0
    for (int i = 0; i < n; i++) {
        x[i] = 0.0; // Initial guess
        r[i] = b[i];
        p[i] = r[i];
    }

    double rsOld = 0.0;
    for (int i = 0; i < n; i++) {
        rsOld += r[i] * r[i];
    }

    for (int iter = 0; iter < maxIter; iter++) {
        // Compute Ap
        for (int i = 0; i < n; i++) {
            Ap[i] = 0.0;
            for (int j = 0; j < n; j++) {
                Ap[i] += A[i][j] * p[j];
            }
        }

        // Compute alpha = rsOld / (p^T Ap)
        double pAp = 0.0;
        for (int i = 0; i < n; i++) {
            pAp += p[i] * Ap[i];
        }
        double alpha = rsOld / pAp;

        // Update x and r
        for (int i = 0; i < n; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        // Check for convergence
        double rsNew = 0.0;
        for (int i = 0; i < n; i++) {
            rsNew += r[i] * r[i];
        }
        if (sqrt(rsNew) < tol) {
            break;
        }

        // Update p
        double beta = rsNew / rsOld;
        for (int i = 0; i < n; i++) {
            p[i] = r[i] + beta * p[i];
        }

        rsOld = rsNew;
    }

    delete[] r;
    delete[] p;
    delete[] Ap;
}
