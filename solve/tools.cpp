#include <cmath>
#include <iostream>


using namespace std;


void jacobiSolver(double* A, double* b, double* x, int size, int maxIterations, double tolerance) {
    double* x_new = (double*)malloc(size * sizeof(double));  // Allocate memory for the updated solution

    for (int i = 0; i < size; i++) {
        x_new[i] = 0.0; // Initialize new solution array
    }

    for (int iter = 0; iter < maxIterations; iter++) {
        for (int i = 0; i < size; i++) {
            double sum = 0.0;
            for (int j = 0; j < size; j++) {
                if (j != i) {
                    sum += A[i* size + j] * x[j];
                }
            }
            x_new[i] = (b[i] - sum) / A[i * size + i];
        }

        // Check for convergence
        double maxDiff = 0.0;
        for (int i = 0; i < size; i++) {
            maxDiff = std::max(maxDiff, std::abs(x_new[i] - x[i]));
            x[i] = x_new[i]; // Update x with x_new
        }

        if (maxDiff < tolerance) {
            break;
        }
    }

    free(x_new); // Free memory for the temporary solution
}

