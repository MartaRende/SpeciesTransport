#include <cmath>
#include <iostream>


using namespace std;


double** invertMatrix(double** A, int n) {
  
    // Create an identity matrix
    double** I = new double*[n];
    for (int i = 0; i < n; i++) {
        I[i] = new double[n];
        for (int j = 0; j < n; j++) {
            I[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Create a copy of A
    double** A_copy = new double*[n];
    for (int i = 0; i < n; i++) {
        A_copy[i] = new double[n];
        for (int j = 0; j < n; j++) {
            A_copy[i][j] = A[i][j];
        }
    }


    // Perform Gauss-Jordan elimination
    for (int i = 0; i < n; i++) {
        // Find pivot
        int pivot = i;
        for (int j = i + 1; j < n; j++) {
            if (std::abs(A_copy[j][i]) > std::abs(A_copy[pivot][i])) {
                pivot = j;
            }
        }

        // Swap rows
        if (pivot != i) {
            std::swap(A_copy[i], A_copy[pivot]);
            std::swap(I[i], I[pivot]);
        }
         
        // Scale row
        double scale = A_copy[i][i];
        for (int j = 0; j < n; j++) {
            A_copy[i][j] /= scale;
            I[i][j] /= scale;
        }


        // Eliminate
        for (int j = 0; j < n; j++) {
            if (j != i) {
                double factor = A_copy[j][i];
                for (int k = 0; k < n; k++) {
                    A_copy[j][k] -= factor * A_copy[i][k];
                    I[j][k] -= factor * I[i][k];
                }
            }
        }
    }

    // Clean up A_copy
    for (int i = 0; i < n; i++) {
        delete[] A_copy[i];
    }
    delete[] A_copy;

    return I;
}


