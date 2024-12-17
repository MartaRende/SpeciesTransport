#include <iostream>
#include <cassert>
#include <cuda_runtime.h>
#include <cmath>
#include "../solve/tools.h"
#include "unitTest.h"
#include <cmath>  // For std::fabs

const double TOLERANCE = 1e-6;


void runTestfillMatrixA(int nx, int ny, double dx, double dy, double D, double dt, const char* testName) {
    size_t values_size = 5 * (nx * ny) * sizeof(double);
    size_t indices_size = 5 * (nx * ny) * sizeof(int);

    double *d_values, *h_values;
    int *d_column_indices, *h_column_indices;
    int *d_row_offsets, *h_row_offsets;

    cudaMalloc(&d_values, values_size);
    cudaMalloc(&d_column_indices, indices_size);
    cudaMalloc(&d_row_offsets, indices_size);

    h_values = (double*)malloc(values_size);
    h_column_indices = (int*)malloc(indices_size);
    h_row_offsets = (int*)malloc(indices_size);

    dim3 blockSize(16, 16);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y);

    initializeRowOffsetsKernel<<<gridSize, blockSize>>>(d_row_offsets, nx, ny);

    fillMatrixAKernel<<<gridSize, blockSize>>>(d_values, d_column_indices, d_row_offsets, dx, dy, D, dt, nx, ny);

    cudaMemcpy(h_values, d_values, values_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_column_indices, d_column_indices, indices_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_row_offsets, d_row_offsets, indices_size, cudaMemcpyDeviceToHost);

    // Perform assertions
  
    // Perform assertions and checks
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            int idx = i * nx + j;
            int row_start = h_row_offsets[idx];

            double expected_diag_value = 1 + dt * D * (2 / (dx * dx) + 2 / (dy * dy));

            if (fabs(h_values[row_start] - expected_diag_value) >= TOLERANCE) {
                std::cerr << "Assertion failed for diagonal value at (" << i << "," << j << "). Expected: " << expected_diag_value
                          << ", Got: " << h_values[row_start] << std::endl;
                exit(EXIT_FAILURE);  // Exit on failure, or you can choose to continue testing
            }

            assert(h_column_indices[row_start] == idx);

            int count = 1;
            double expected_neighbor_value = -dt * D / (dx * dx);

            // Left neighbor
            if (fabs(h_values[row_start + count] - expected_neighbor_value) >= TOLERANCE) {
                std::cerr << "Assertion failed for left neighbor at (" << i << "," << j << "). Expected: " << expected_neighbor_value
                          << ", Got: " << h_values[row_start + count] << std::endl;
                exit(EXIT_FAILURE);
            }
            assert(h_column_indices[row_start + count] == idx - 1);
            count++;

            // Right neighbor
            if (fabs(h_values[row_start + count] - expected_neighbor_value) >= TOLERANCE) {
                std::cerr << "Assertion failed for right neighbor at (" << i << "," << j << "). Expected: " << expected_neighbor_value
                          << ", Got: " << h_values[row_start + count] << std::endl;
                exit(EXIT_FAILURE);
            }
            assert(h_column_indices[row_start + count] == idx + 1);
            count++;

            expected_neighbor_value = -dt * D / (dy * dy);

            // Top neighbor
            if (fabs(h_values[row_start + count] - expected_neighbor_value) >= TOLERANCE) {
                std::cerr << "Assertion failed for top neighbor at (" << i << "," << j << "). Expected: " << expected_neighbor_value
                          << ", Got: " << h_values[row_start + count] << std::endl;
                exit(EXIT_FAILURE);
            }
            assert(h_column_indices[row_start + count] == idx - nx);
            count++;

            // Bottom neighbor
            if (fabs(h_values[row_start + count] - expected_neighbor_value) >= TOLERANCE) {
                std::cerr << "Assertion failed for bottom neighbor at (" << i << "," << j << "). Expected: " << expected_neighbor_value
                          << ", Got: " << h_values[row_start + count] << std::endl;
                exit(EXIT_FAILURE);
            }
            assert(h_column_indices[row_start + count] == idx + nx);
            count++;
        }
    }

    cudaFree(d_values);
    cudaFree(d_column_indices);
    cudaFree(d_row_offsets);
    free(h_values);
    free(h_column_indices);
    free(h_row_offsets);

    std::cout << testName << " passed successfully." << std::endl;
}
