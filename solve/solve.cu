#include "solve.h"
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include "../common_includes.c"

// #include "tools.h"

#include <chrono>
using namespace std;
using namespace chrono;

void computeBoundaries(double **Y, const int nx, const int ny)
{
    for (int i = 0; i < nx; i++)
    {
        Y[i][ny - 1] = 0.0;
        Y[i][0] = 0.0;
    }

    for (int j = 0; j < ny; j++)
    {
        Y[0][j] = 0.0;
        Y[nx - 1][j] = 0.0;
    }
}
__global__ void fillMatrixAKernel(double *A, const double dx, const double dy, const double D, const double dt, const int nx, const int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

     int totSize = nx * ny;

    A[i] = 0.0; // Initialize with zero

    // Populate internal nodes

    if (i > 0 && i < (nx - 1) && j > 0 && j < (ny - 1))
    {
        int idx = i * ny + j;

        // Diagonal (central node)
        A[idx * totSize + idx] = 1 + dt * D * (2 / (dx * dx) + 2 / (dy * dy));

        // Left neighbor
        A[idx * totSize + (idx - 1)] = -dt * D / (dx * dx);

        // Right neighbor
        A[idx * totSize + (idx + 1)] = -dt * D / (dx * dx);

        // Top neighbor
        A[idx * totSize + ((i - 1) * ny + j)] = -dt * D / (dy * dy);

        // Bottom neighbor
        A[idx * totSize + ((i + 1) * ny + j)] = -dt * D / (dy * dy);
    }

    // Handle boundary conditions

    // Bottom boundary
    int idxBottom = i * ny;
    A[idxBottom * totSize + idxBottom] = 1.0;

    // Top boundary
    int idxTop = i * ny + (ny - 1);
    A[idxTop * totSize + idxTop] = 1.0;

    // Left boundary
    int idxLeft = j;
    A[idxLeft * totSize + idxLeft] = 1.0;

    // Right boundary
    int idxRight = (nx - 1) * ny + j;
    A[idxRight * totSize + idxRight] = 1.0;
   
}

__global__ void computeB(double *b, double* Y_n,double *u, double *v, const double dx, const double dy, const int nx, const int ny, const double dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i == 0 || i == (nx - 1))
    {
        return;
    }
    if (j == 0 || j == (ny - 1))
    {
        return;
    }

    int idx = i * ny + j;
    int right = i * ny + (j - 1);
    int left = i * ny + (j + 1);
    int top = (i - 1) * ny + j;
    int down = (i + 1) * ny + j;

    b[idx] = Y_n[idx];

    if (u[idx] < 0.0)
    {
        b[idx] -= dt * (u[idx] * (Y_n[down] - Y_n[idx]) / dx);
    }
    else
    {
        b[idx] -= dt * (u[idx] * (Y_n[idx] - Y_n[top]) / dx);
    }

    if (v[idx] < 0.0)
    {
        b[idx] -= dt * (v[idx] * (Y_n[right] - Y_n[idx]) / dy);
    }
    else
    {
        b[idx] -= dt * (v[idx] * (Y_n[idx] - Y_n[left]) / dy);
    }
}
void solveSpeciesEquation(double **Y, double **u, double **v, const double dx, const double dy, double D, const int nx, const int ny, const double dt)
{
    auto start_total_solve = high_resolution_clock::now();
    size_t unidimensional_size_bytes = nx * ny * sizeof(double);

    // Allocate memory using malloc
    double *Y_n = (double *)malloc(unidimensional_size_bytes);       // Previous Y
    double *x = (double *)malloc(unidimensional_size_bytes);         // Flattened solution vector
    double *A = (double *)malloc(unidimensional_size_bytes * nx *ny);        // Matrix A (flattened to 2D)
    double *b_flatten = (double *)malloc(unidimensional_size_bytes); // Flattened b vector
    double *u_flatten = (double *)malloc(unidimensional_size_bytes);
    double *v_flatten = (double *)malloc(unidimensional_size_bytes);

    double *d_Yn;
    double *d_x;
    double *d_A;
    double *d_b_flatten;
    double *d_u;
    double *d_v;

    CHECK_ERROR(cudaMalloc((void **)&d_Yn, unidimensional_size_bytes));
    CHECK_ERROR(cudaMalloc((void **)&d_x, unidimensional_size_bytes));
    CHECK_ERROR(cudaMalloc((void **)&d_A, unidimensional_size_bytes *nx*ny));
    CHECK_ERROR(cudaMalloc((void **)&d_b_flatten, unidimensional_size_bytes));
    CHECK_ERROR(cudaMalloc((void **)&d_u, unidimensional_size_bytes));
    CHECK_ERROR(cudaMalloc((void **)&d_v, unidimensional_size_bytes));

    for (int i = 0; i < nx * ny; ++i)
    {   
        x[i] = 0.0;    
    }

    for (int i = 0; i < nx; i++)
    {

        for (int j = 0; j < ny; j++)
        {
            Y_n[i * ny + j] = Y[i][j];
            u_flatten[j * ny + i] = u[i][j];
            v_flatten[j * ny + i] = v[i][j];
        }
    }
    CHECK_ERROR(cudaMemcpy(d_u, u_flatten, unidimensional_size_bytes, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(d_v, v_flatten, unidimensional_size_bytes, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(d_Yn, Y_n, unidimensional_size_bytes, cudaMemcpyHostToDevice));

    auto end_init_solve = duration_cast<microseconds>(high_resolution_clock::now() - start_total_solve).count();
    printf("[SOLVE] Initialization took: %ld us\n", end_init_solve);

    dim3 blockDim(10, 10);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y);
    // Compute part Ax = b
    // Fill A with right coefficients.
    auto start_fillMatrix = high_resolution_clock::now();

    fillMatrixAKernel<<<gridDim, blockDim>>>(d_A, dx, dy, D, dt, nx, ny);
    auto end_fillMatrix = duration_cast<microseconds>(high_resolution_clock::now() - start_fillMatrix).count();
    printf("[SOLVE] Fill Matrix A took: %ld us\n", end_fillMatrix);

    // Compute b part
    auto start_fillb = high_resolution_clock::now();
    computeB<<<gridDim, blockDim>>>(d_b_flatten, d_Yn,d_u, d_v, dx, dy, nx, ny, dt);
    cudaDeviceSynchronize();
    CHECK_ERROR(cudaMemcpy(A, d_A, unidimensional_size_bytes*nx*ny, cudaMemcpyDeviceToHost));
    CHECK_ERROR(cudaMemcpy(b_flatten, d_b_flatten, unidimensional_size_bytes, cudaMemcpyDeviceToHost));
    
    auto end_fillb = duration_cast<microseconds>(high_resolution_clock::now() - start_fillb).count();
    printf("[SOLVE] Fill b took: %ld us\n", end_fillb);

    // Compute x with an iterative method
    auto start_computex = high_resolution_clock::now();
    jacobiSolver(A, b_flatten, x, nx * ny, 1000, 1e-2);
    auto end_computex = duration_cast<microseconds>(high_resolution_clock::now() - start_computex).count();
    printf("[SOLVE] Fill x took: %ld us\n", end_computex);

    // Update Yi
    for (int i = 1; i < nx - 1; i++)
    {
        for (int j = 1; j < ny - 1; j++)
        {
            Y[i][j] = x[i * ny + j];
           
        }
    }

    computeBoundaries(Y, nx, ny);
    cudaFree(d_Yn);
    cudaFree(d_x);
    cudaFree(d_A);
    cudaFree(d_b_flatten);
    cudaFree(d_u);
    cudaFree(d_v);

    // Free memory using free()
   
    free(A); // Free the pointer to the array of rows
    free(Y_n); // Free the pointer to the array of rows

    free(x);         // Free the flattened solution vector
    free(b_flatten); // Free the flattened b vector

    auto end_total_solve = duration_cast<microseconds>(high_resolution_clock::now() - start_total_solve).count();
    printf("[SOLVE] Total time taken: %ld us\n", end_total_solve);
}
