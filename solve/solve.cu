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


void solveSpeciesEquation(double *Y, 
                          const double dx, const double dy, double D,
                          const int nx, const int ny, const double dt,double * d_u, double * d_v, double * d_Yn, double * d_x, double * d_x_new, double * d_b_flatten, double * d_values, int * d_column_indices, int * d_row_offsets)
{
    auto start_total_solve = high_resolution_clock::now();

    int max_iter = 1;
    double tol = 1e-3;
    size_t unidimensional_size_of_bytes = nx * ny * sizeof(double);
    size_t nnz_estimate = nx * ny * 5;

    // Allocate host memory
    double *x = (double *)malloc(unidimensional_size_of_bytes);
    double *b_flatten = (double *)malloc(unidimensional_size_of_bytes);
    /* for(int i = 0; i<nx*ny;i++){
        printf("%f\n",Y[i]);
    } */

    // Copy input data to device
    CHECK_ERROR(cudaMemcpy(d_Yn, Y, unidimensional_size_of_bytes, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(d_x, d_Yn, unidimensional_size_of_bytes, cudaMemcpyDeviceToDevice));
    cudaMemset(d_x_new, 0, nx * ny * sizeof(double));

    dim3 blockDim(16, 16);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y);


    auto end_init_solve = duration_cast<microseconds>(high_resolution_clock::now() - start_total_solve).count();
    printf("[SOLVE] Initialization took: %ld us\n", end_init_solve);
    auto start_fillMatrix = high_resolution_clock::now();

    // Fill A
initializeRowOffsetsKernel<<<gridDim,blockDim>>>(d_row_offsets,nx,ny);
 //cudaDeviceSynchronize();

    fillMatrixAKernel<<<gridDim, blockDim>>>(d_values, d_column_indices, d_row_offsets, dx, dy, D, dt, nx, ny);
    auto end_fillMatrix = duration_cast<microseconds>(high_resolution_clock::now() - start_fillMatrix).count();
    printf("[SOLVE] Fill Matrix A took: %ld us\n", end_fillMatrix);
    // cudaDeviceSynchronize();

    // Compute b
    auto start_fillb = high_resolution_clock::now();
    computeB<<<gridDim, blockDim>>>(d_b_flatten, d_Yn, d_u, d_v, dx, dy, nx, ny, dt);

   // cudaDeviceSynchronize();
//    CHECK_ERROR(cudaMemcpy(b_flatten, d_b_flatten, unidimensional_size_of_bytes, cudaMemcpyDeviceToHost));

    auto end_fillb = duration_cast<microseconds>(high_resolution_clock::now() - start_fillb).count();
       printf("[SOLVE] Fill b took: %ld us\n", end_fillb);

    auto start_computex = high_resolution_clock::now();

    // Jacobi Solver
    for (int iter = 0; iter < 1; ++iter)
    {
        // Launch Jacobi kernel
        jacobiKernel<<<gridDim, blockDim>>>(d_row_offsets, d_column_indices, d_values, d_b_flatten, d_x, d_x_new, nx, ny, 5 * nx * ny);
        //cudaDeviceSynchronize();
        printf("here");
        // Launch difference kernel
  //     dim3 blockDim(256);  // Or any other suitable block size
//dim3 gridDim((nx * ny + blockDim.x - 1) / blockDim.x);
//diffKernel<<<gridDim, blockDim>>>(d_x, d_x_new, d_diff, nx, ny);
    
     //   cudaDeviceSynchronize();
    cudaMemcpy(d_x, d_x_new, nx * ny * sizeof(double), cudaMemcpyDeviceToDevice);

    
    }

    auto end_computex = duration_cast<microseconds>(high_resolution_clock::now() - start_computex).count();
    printf("[SOLVE] Fill x took: %ld us\n", end_computex);

computeBoundariesKernel<<<gridDim, blockDim>>>(d_x, nx, ny);

    // Copy results back to host
    CHECK_ERROR(cudaMemcpy(Y, d_x, unidimensional_size_of_bytes, cudaMemcpyDeviceToHost));

    // Update Y
  

   

    // Free host memory
    free(x);
    free(b_flatten);

    auto end_total_solve = duration_cast<microseconds>(high_resolution_clock::now() - start_total_solve).count();
    printf("[SOLVE] Total time taken: %ld us\n", end_total_solve);
}
__global__ void computeBoundariesKernel(double *Y, const int nx, const int ny)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < ny && j < nx) {
        // Left and right boundaries
        if (j == 0 || j == nx - 1) {
            Y[i * nx + j] = 0.0;
        }
        
        // Top and bottom boundaries
        if (i == 0 || i == ny - 1) {
            Y[i * nx + j] = 0.0;
        }
    }
}
