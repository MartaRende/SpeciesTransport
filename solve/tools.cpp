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

void fillMatrixA(double **A, const double dx, const double dy, const double D, const double dt, const int nx, const int ny)
{
    for (int i = 0; i < nx * ny; ++i)
    {
        for (int j = 0; j < nx * ny; ++j)
        {
            A[i][j] = 0.0; 
        }
    }

    for (int i = 1; i < nx - 1; ++i)
    {
        for (int j = 1; j < ny - 1; ++j)
        {
            int idx = i * ny + j;

            A[idx][idx] = 1 + dt * D * (2 / (dx * dx) + 2 / (dy * dy));

            A[idx][idx - 1] = -dt * D / (dx * dx);

            A[idx][idx + 1] = -dt * D / (dx * dx);

            A[idx][(i - 1) * ny + j] = -dt * D / (dy * dy);

            A[idx][(i + 1) * ny + j] = -dt * D / (dy * dy);
        }
    }

    for (int i = 0; i < nx; ++i)
    {
        // Bottom boundary
        A[i * ny][i * ny] = 1.0;

        // Top boundary
        A[i * ny + (ny - 1)][i * ny + (ny - 1)] = 1.0;
    }
    for (int j = 0; j < ny; ++j)
    {
        // Left boundary
        A[j][j] = 1.0;

        // Right boundary
        A[(nx - 1) * ny + j][(nx - 1) * ny + j] = 1.0;
    }
}