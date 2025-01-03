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

void solveSpeciesEquation(
                          const double dx, const double dy, double D,
                          const int nx, const int ny, const double dt, double *d_u, double *d_v, double *d_Yn, double *d_x, double *d_x_new, double *d_b, double *d_values, int *d_column_indices, int *d_row_offsets, int world_rank)
{

    /*It's important to ensure that dx and dy are positive because otherwise
    we would generate calculations with divisions by zero  */

    if (dx < 0 || dy < 0)
    {
        std::cerr << "dx and dy must be positive " << std::endl;
        std::exit(-1);
    }

    // == Start solve part ==
    auto start_total_solve = high_resolution_clock::now();

    // == Paramenters to solve jacobi iteration method ==
    int max_iter = 100;
    double tol = 1e-20; // the smaller it is the more accurate it will be 

    size_t unidimensional_size_of_bytes = nx * ny * sizeof(double);

    cudaMemset(d_x_new, 0, nx * ny * sizeof(double));

    //== all kernel are 2d and they have the same size ==
    dim3 blockDim(16, 16);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y);

    auto end_init_solve = duration_cast<microseconds>(high_resolution_clock::now() - start_total_solve).count();
    auto start_fillMatrix = high_resolution_clock::now();

    // == Part to fill matrix A --> diffusion part  ==

    initializeRowOffsetsKernel<<<gridDim, blockDim>>>(d_row_offsets, nx, ny); // row offset calculation

    fillMatrixAKernel<<<gridDim, blockDim>>>(d_values, d_column_indices, d_row_offsets, dx, dy, D, dt, nx, ny); // Calculate the non-zero values and column indices of A 

    auto end_fillMatrix = duration_cast<microseconds>(high_resolution_clock::now() - start_fillMatrix).count();

    // == Compute b part --> advection part ==

    auto start_fillb = high_resolution_clock::now();
    computeB<<<gridDim, blockDim>>>(d_b, d_Yn, d_u, d_v, dx, dy, nx, ny, dt);


    cudaDeviceSynchronize(); // this is necessary to be sure that the values of A and b are finished to be calculated because if not, we may have errors in the calculation of the system Ax = b

    auto end_fillb = duration_cast<microseconds>(high_resolution_clock::now() - start_fillb).count();

    // == beginning of the calculation of x using jacobi's iterative method ==
    auto start_computex = high_resolution_clock::now();

    jacobiKernel<<<gridDim, blockDim>>>(d_row_offsets, d_column_indices, d_values, d_b, d_x, d_x_new, nx, ny, 5 * nx * ny, max_iter, tol);

    auto end_computex = duration_cast<microseconds>(high_resolution_clock::now() - start_computex).count();

    computeBoundariesKernel<<<gridDim, blockDim>>>(d_x, nx, ny); 

    cudaDeviceSynchronize(); // wait until the iterative method has finished before copying the values 

    // == copy de actual value of my species d_x into d_Yn for the next iteration
    cudaMemcpy(d_Yn, d_x, unidimensional_size_of_bytes, cudaMemcpyDeviceToDevice); // less expensive to make a copy from device to device than from device to host

    auto end_total_solve = duration_cast<microseconds>(high_resolution_clock::now() - start_total_solve).count();
    // print the time taken by each part
    if (world_rank == 0)
    {
        printf("[SOLVE] Initialization took: %ld us\n", end_init_solve);

        printf("[SOLVE] Fill Matrix A took: %ld us\n", end_fillMatrix);

        printf("[SOLVE] Fill b took: %ld us\n", end_fillb);

        printf("[SOLVE] Fill x took: %ld us\n", end_computex);

        printf("[SOLVE] Total time taken: %ld us\n", end_total_solve);
    }
}

