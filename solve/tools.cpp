#include <cmath>
#include <iostream>

#include <Eigen/Dense>

using namespace Eigen;
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
double determinant(double** A, int n) {
    double det = 1.0;
    double** matrix = new double*[n]; // Create a copy of the matrix
    for (int i = 0; i < n; i++) {
        matrix[i] = new double[n];
        for (int j = 0; j < n; j++) {
            matrix[i][j] = A[i][j]; // Copy elements of A to matrix
        }
    }

    // Perform Gaussian Elimination
    for (int i = 0; i < n; ++i) {
        // Find the pivot element
        double pivot = matrix[i][i];
        if (pivot == 0) {
            // If pivot is zero, determinant is zero (matrix is singular)
            for (int i = 0; i < n; i++) {
                delete[] matrix[i]; // Free memory
            }
            delete[] matrix;
            return 0;
        }

        // Scale the matrix so that the pivot is 1
        for (int j = i + 1; j < n; ++j) {
            double scale = matrix[j][i] / pivot;
            for (int k = i; k < n; ++k) {
                matrix[j][k] -= scale * matrix[i][k];
            }
        }

        // Multiply the pivot element into the determinant
        det *= pivot;
    }

    // The determinant is the product of the diagonal elements
    for (int i = 0; i < n; ++i) {
        delete[] matrix[i]; // Free memory
    }
    delete[] matrix;
    return det;
}

MatrixXd computePseudoinverse(MatrixXd& A) {
    // Perform SVD
    JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);
    MatrixXd U = svd.matrixU();
    MatrixXd V = svd.matrixV();
    VectorXd S = svd.singularValues();

    // Compute Sigma^+
    MatrixXd Sigma_pseudo = MatrixXd::Zero(A.cols(), A.rows());
    for (int i = 0; i < S.size(); ++i) {
        if (S(i) > 1e-10) { // Avoid division by zero
            Sigma_pseudo(i, i) = 1.0 / S(i);
        }
    }

    // Compute pseudoinverse: A^+ = V * Sigma^+ * U^T
    return V * Sigma_pseudo * U.transpose();
}
