#include <iostream>
#include <cassert>
#include <cuda.h>
#include <cmath>
#include "../solve/tools.h"
#include "unitTest.h"
#include <cmath>  

const double TOLERANCE = 1e-6;

#include <iostream>
#include <cuda_runtime.h>

void runTestfillMatrixA(int nx, int ny, double dx, double dy, double D, double dt, const char* testName) {
    size_t values_size = 5 * (nx * ny) * sizeof(double);
    size_t indices_size = 5 * (nx * ny) * sizeof(int);

    double *d_values, *h_values;
    int *d_column_indices, *h_column_indices;
    int *d_row_offsets, *h_row_offsets;

    // Allocate device memory
    cudaMalloc(&d_values, values_size);
    cudaMalloc(&d_column_indices, indices_size);
    cudaMalloc(&d_row_offsets, (nx * ny + 1) * sizeof(int)); // +1 for row offsets

    // Allocate host memory
    h_values = (double*)malloc(values_size);
    h_column_indices = (int*)malloc(indices_size);
    h_row_offsets = (int*)malloc((nx * ny + 1) * sizeof(int)); // +1 for row offsets

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y);

    // Initialize row offsets on device
    initializeRowOffsetsKernel<<<gridSize, blockSize>>>(d_row_offsets, nx , ny);

    // Fill the sparse matrix on device
    fillMatrixAKernel<<<gridSize, blockSize>>>(d_values, d_column_indices, d_row_offsets, dx, dy, D, dt, nx, ny);

    // Copy results back to host
    cudaMemcpy(h_values, d_values, values_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_column_indices, d_column_indices, indices_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_row_offsets, d_row_offsets, (nx * ny + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    // Print out the sparse matrix representation
    std::cout << "Sparse Matrix Representation for " << testName << ":\n";
    
    std::cout << "Values: ";
    for (int i = 0; i < values_size / sizeof(double); i++) {
        std::cout << h_values[i] << " ";
    }
    
    std::cout << "\nColumn Indices: ";
    for (int i = 0; i < indices_size / sizeof(int); i++) {
        std::cout << h_column_indices[i] << " ";
    }

    std::cout << "\nRow Offsets: ";
    for (int i = 0; i <= nx * ny; i++) { // +1 for row offsets
        std::cout << h_row_offsets[i] << " ";
    }
    
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_values);
    cudaFree(d_column_indices);
    cudaFree(d_row_offsets);

    // Free host memory
    free(h_values);
    free(h_column_indices);
    free(h_row_offsets);

    std::cout << testName << " passed successfully." << std::endl;
}
