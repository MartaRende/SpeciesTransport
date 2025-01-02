#include <stdio.h>
#include <iostream>
#include <unordered_map>
#include "../common_includes.c"
#include <chrono>
using namespace std;
using namespace chrono;
#include <vector>
#include <cuda.h>
#include "solve.h"
#include "tools.h"
#include <cstdlib> // Per std::exit

void solveSpeciesEquation(double *Y,
                          const double dx, const double dy, double D,
                          const int nx, const int ny, const double dt, double *d_u, double *d_v, double *d_Yn, double *d_x, double *d_x_new, double *d_b_flatten, double *d_values, int *d_column_indices, int *d_row_offsets, int world_rank)
{

    /*It's important to ensure that dx and dy are positive because otherwise
    we would generate calculations with divisions by zero  */

    if (dx < 0 || dy < 0)
    {
        std::cerr << "dx and dy must be positive " << std::endl;
        std::exit(-1);
    }

    // start solve part
    auto start_total_solve = high_resolution_clock::now();

    int max_iter = 100;
    double tol = 1e-20;
    size_t unidimensional_size_of_bytes = nx * ny * sizeof(double);
    size_t nnz_estimate = nx * ny * 5;

    cudaMemset(d_x_new, 0, nx * ny * sizeof(double));
    dim3 blockDim(16, 16);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y);

    auto end_init_solve = duration_cast<microseconds>(high_resolution_clock::now() - start_total_solve).count();
    auto start_fillMatrix = high_resolution_clock::now();

    // Fill A
    initializeRowOffsetsKernel<<<gridDim, blockDim>>>(d_row_offsets, nx, ny);
    cudaDeviceSynchronize();

    fillMatrixAKernel<<<gridDim, blockDim>>>(d_values, d_column_indices, d_row_offsets, dx, dy, D, dt, nx, ny);
    auto end_fillMatrix = duration_cast<microseconds>(high_resolution_clock::now() - start_fillMatrix).count();

    // Compute b
    auto start_fillb = high_resolution_clock::now();
    computeB<<<gridDim, blockDim>>>(d_b_flatten, d_Yn, d_u, d_v, dx, dy, nx, ny, dt);

    // cudaDeviceSynchronize();

    auto end_fillb = duration_cast<microseconds>(high_resolution_clock::now() - start_fillb).count();

    auto start_computex = high_resolution_clock::now();

    jacobiKernel<<<gridDim, blockDim>>>(d_row_offsets, d_column_indices, d_values, d_b_flatten, d_x, d_x_new, nx, ny, 5 * nx * ny, max_iter, tol);

    auto end_computex = duration_cast<microseconds>(high_resolution_clock::now() - start_computex).count();

    computeBoundariesKernel<<<gridDim, blockDim>>>(d_x, nx, ny);

    cudaMemcpy(d_Yn, d_x, unidimensional_size_of_bytes, cudaMemcpyDeviceToDevice);

    // Copy results back to host

    auto end_total_solve = duration_cast<microseconds>(high_resolution_clock::now() - start_total_solve).count();
    if (world_rank == 0)
    {
        printf("[SOLVE] Initialization took: %ld us\n", end_init_solve);

        printf("[SOLVE] Fill Matrix A took: %ld us\n", end_fillMatrix);

        printf("[SOLVE] Fill b took: %ld us\n", end_fillb);

        printf("[SOLVE] Fill x took: %ld us\n", end_computex);

        printf("[SOLVE] Total time taken: %ld us\n", end_total_solve);
    }
}
__global__ void computeBoundariesKernel(double *Y, const int nx, const int ny)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * nx + j;
    if (i < ny && j < nx)
    {
        if (j == 0 || j == nx - 1 || i == 0 || i == ny - 1)
        {
            Y[idx] = 0.0;
        }
    }
}
