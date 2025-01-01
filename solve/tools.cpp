#include <cmath>
#include <iostream>
#include "solve.h"
using namespace std;


void jacobiSolver(SparseMatrix &A_sparse, double *b, double *x, int n, int max_iter, double tol)
{
    double *x_new = new double[n];
    for (int iter = 0; iter < max_iter; ++iter)
    {
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
       {
            break;
       }
        // Update x
        for (int i = 0; i < n; ++i)
            x[i] = x_new[i];
    }
    delete[] x_new;
}

