#include <cmath>
#include <iostream>


using namespace std;


void jacobiSolver(double** A, double* b, double* x, int size, int maxIterations, double tolerance) {
    double* x_new = new double[size]; // Allocate memory for the updated solution

    for (int i = 0; i < size; i++) {
        x_new[i] = 0.0; // Initialize new solution array
    }

    for (int iter = 0; iter < maxIterations; iter++) {
        for (int i = 0; i < size; i++) {
            double sum = 0.0;
            for (int j = 0; j < size; j++) {
                if (j != i) {
                    sum += A[i][j] * x[j];
                }
            }
            x_new[i] = (b[i] - sum) / A[i][i];
        }

        // Check for convergence
        double maxDiff = 0.0;
        for (int i = 0; i < size; i++) {
            maxDiff = std::max(maxDiff, std::abs(x_new[i] - x[i]));
            x[i] = x_new[i]; // Update x with x_new
        }

        if (maxDiff < tolerance) {
            //printf("[JACOBI] Converged in %d iterations with tolerance: %.6e\n", iter + 1, tolerance);
            break;
        }
    }

    delete[] x_new; // Free memory for the temporary solution
}

