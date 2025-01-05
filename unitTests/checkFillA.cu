#include <iostream>
#include <cassert>
#include <cuda.h>
#include <cmath>
#include "../solve/tools.h"
#include "unitTest.h"
#include <cmath>


void runTestfillMatrixA(int* row_offset,double* exp_values,int nx, int ny, double dx, double dy, double D, double dt, const char *testName )
{
    //Chcking invalid cases
    if (dx <= 0 || dy <= 0)
    {
throw std::invalid_argument("dx and dy must be positive.");    }

    size_t values_size = 5 * (ny) * sizeof(double);
    size_t indices_size = 5 * (ny) * sizeof(int);

    double *d_values, *h_values;
    int *d_column_indices, *h_column_indices;
    int *d_row_offsets;

    cudaMalloc(&d_row_offsets, (ny+1) * sizeof(int));
    cudaMemcpy(d_row_offsets, row_offset, (ny+1) * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate device memory
    cudaMalloc(&d_values, values_size);
    cudaMalloc(&d_column_indices, indices_size);

    // Allocate host memory
    h_values = (double *)malloc(values_size);
    h_column_indices = (int *)malloc(indices_size);
    // h_row_offsets = (int*)malloc((nx * ny + 1) * sizeof(int)); // +1 for row offsets

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y);



    // Fill the sparse matrix on device
    fillMatrixAKernel<<<gridSize, blockSize>>>(d_values, d_column_indices, d_row_offsets, dx, dy, D, dt, nx, ny);

    // Copy results back to host
    cudaMemcpy(h_values, d_values, values_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_column_indices, d_column_indices, indices_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(row_offset, d_row_offsets, (ny+1) * sizeof(int), cudaMemcpyDeviceToHost);

    double tolerance = 0.0001;  // tolerence for double values

    for (int i = 0; i < values_size / sizeof(double); i++)
    {
        assert(fabs(h_values[i] - exp_values[i]) < tolerance);

    }

    
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_values);
    cudaFree(d_column_indices);
    cudaFree(d_row_offsets);

    // Free host memory
    free(h_values);
    free(h_column_indices);

    std::cout << testName << " passed successfully." << std::endl;
}
