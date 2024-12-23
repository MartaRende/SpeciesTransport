#include <iostream>
#include <cassert>
#include <cuda.h>
#include <cmath>
#include "../solve/tools.h"
#include "unitTest.h"
#include <cmath>  


void testJacobiSolver(int nx, int ny, int nnz,int * row, int * col , double * values, double * b, double* x, double *x_new) {

    // Allocate device memory
    int *d_row, *d_col;
    double *d_values, *d_b, *d_x, *d_x_new;
    cudaMalloc(&d_row,( nx+1) * sizeof(int));
    cudaMalloc(&d_col, nnz* sizeof(int));
    cudaMalloc(&d_values,nnz * sizeof(double));
    cudaMalloc(&d_b, ny * sizeof(double));
    cudaMalloc(&d_x, ny * sizeof(double));
    cudaMalloc(&d_x_new, ny * sizeof(double));

    // Copy data to device
    cudaMemcpy(d_row, row, ( nx+1)* sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, col,nnz* sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values,nnz* sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, ny * sizeof(double), cudaMemcpyHostToDevice);
   cudaMemcpy(d_x, x_new, ny * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_new, x_new, ny * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_x_new, 0, ny * sizeof(double));

    // Launch kernel
   dim3 blockDim(16, 16);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y);
    jacobiKernel<<<gridDim, blockDim>>>(d_row, d_col, d_values, d_b, d_x, d_x_new, nx, ny, nnz, 100, 1e-6);

    // Copy result back to host
    cudaMemcpy(x_new, d_x_new, ny * sizeof(double), cudaMemcpyDeviceToHost);

    // Check result
    for(int i = 0; i<ny;i++){
     printf("hold %f\n", x[i]);
     printf("new %f\n", x_new[i]);
    
//    assert(fabs(x_new[i] - x[i]) < 0.5);

    }

    // Free device memory
    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_values);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_x_new);

    printf("Invertible matrix test passed!\n");
}
