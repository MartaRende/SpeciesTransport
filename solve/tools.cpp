#include <cmath>
#include <iostream>
#include "solve.h"
using namespace std;

// == method to compute system Ax = b ==
void jacobiSolver(SparseMatrix &A_sparse, double *b, double *x, int n, int max_iter, double tol)
{
    double *x_new = new double[n];
    for (int iter = 0; iter < max_iter; ++iter)
    {
        // Compute Jacobi formula
        // Initialize x_new with b
        for (int i = 0; i < n; ++i)
            x_new[i] = b[i];

        // Apply sparse matrix entries
        for (size_t k = 0; k < A_sparse.value.size(); ++k)
        {
            int i = A_sparse.row[k];
            int j = A_sparse.col[k];

            if (i == j) // Diagonal element
                x_new[i] /= A_sparse.value[k];
            else
                x_new[i] -= A_sparse.value[k] * x[j];
        }

        // Check for convergence
        double diff = 0.0;
        for (int i = 0; i < n; ++i)
            diff += abs(x_new[i] - x[i]);

        if (diff < tol)
            break;
    
        // Update x
        for (int i = 0; i < n; ++i)
            x[i] = x_new[i];
    }
    delete[] x_new;
}

// == Fonction to fill Matrix A

void fillMatrixA(SparseMatrix &A_sparse, const double dx, const double dy, const double D, const double dt, const int nx, const int ny)
{
    // Only Internal nodes
    for (int i = 1; i < nx - 1; ++i)
    {
        for (int j = 1; j < ny - 1; ++j)
        {
            int idx = i * ny + j;

            // Diagonal (central node)
            A_sparse.row.push_back(idx);
            A_sparse.col.push_back(idx);
            A_sparse.value.push_back(1 + dt * D * (2 / (dx * dx) + 2 / (dy * dy)));

            // Left neighbor
            A_sparse.row.push_back(idx);
            A_sparse.col.push_back(idx - 1);
            A_sparse.value.push_back(-dt * D / (dx * dx));

            // Right neighbor
            A_sparse.row.push_back(idx);
            A_sparse.col.push_back(idx + 1);
            A_sparse.value.push_back(-dt * D / (dx * dx));

            // Top neighbor
            A_sparse.row.push_back(idx);
            A_sparse.col.push_back((i - 1) * ny + j);
            A_sparse.value.push_back(-dt * D / (dy * dy));

            // Bottom neighbor
            A_sparse.row.push_back(idx);
            A_sparse.col.push_back((i + 1) * ny + j);
            A_sparse.value.push_back(-dt * D / (dy * dy));
        }
    }

}



// boundaries are set to 0 for simplicity
void computeBoundaries(double **Y, const int nx, const int ny)
{
    for (int i = 0; i < nx; i++)
    {
        Y[i][ny - 1] = 0.0;
        Y[i][0] = 0.0;
    }

    for (int j = 0; j < ny; j++)
    {
        Y[0][j] = 0.0;
        Y[nx - 1][j] = 0.0;
    }
}
