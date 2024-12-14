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

__global__ void fillMatrixAKernel(double *values, int *column_indices, int *row_offsets,
                                  const double dx, const double dy, const double D,
                                  const double dt, const int nx, const int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny || i == 0 || i == nx - 1 || j == 0 || j == ny - 1)
        return;

    int idx = i * ny + j;
    int count = 0;

    int row_start = row_offsets[idx];

    // Diagonal
    values[row_start + count] = 1 + dt * D * (2 / (dx * dx) + 2 / (dy * dy));
    column_indices[row_start + count++] = idx;

    // Left Neighbor
    values[row_start + count] = -dt * D / (dx * dx);
    column_indices[row_start + count++] = idx - ny;

    // Right Neighbor
    values[row_start + count] = -dt * D / (dx * dx);
    column_indices[row_start + count++] = idx + ny;

    // Top Neighbor
    values[row_start + count] = -dt * D / (dy * dy);
    column_indices[row_start + count++] = idx - 1;

    // Bottom Neighbor
    values[row_start + count] = -dt * D / (dy * dy);
    column_indices[row_start + count++] = idx + 1;
}

__global__ void computeB(double *b, double *Y_n, double *u, double *v,
                         const double dx, const double dy, const int nx, const int ny, const double dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1)
        return;

    int idx = i * ny + j;
    int right = i * ny + (j + 1);
    int left = i * ny + (j - 1);
    int top = (i - 1) * ny + j;
    int down = (i + 1) * ny + j;

    b[idx] = Y_n[idx];

    if (u[idx] < 0.0)
        b[idx] -= dt * (u[idx] * (Y_n[down] - Y_n[idx]) / dx);
    else
        b[idx] -= dt * (u[idx] * (Y_n[idx] - Y_n[top]) / dx);

    if (v[idx] < 0.0)
        b[idx] -= dt * (v[idx] * (Y_n[right] - Y_n[idx]) / dy);
    else
        b[idx] -= dt * (v[idx] * (Y_n[idx] - Y_n[left]) / dy);
}

void solveSpeciesEquation(double **Y, double **u, double **v,
                          const double dx, const double dy, double D,
                          const int nx, const int ny, const double dt)
{
    auto start_total_solve = high_resolution_clock::now();

    size_t unidimensional_size_bytes = nx * ny * sizeof(double);
    size_t nnz_estimate = nx * ny * 5;

    // Allocate host memory
    double *Y_n = (double *)malloc(unidimensional_size_bytes);
    double *x = (double *)malloc(unidimensional_size_bytes);
    double *b_flatten = (double *)malloc(unidimensional_size_bytes);
    double *u_flatten = (double *)malloc(unidimensional_size_bytes);
    double *v_flatten = (double *)malloc(unidimensional_size_bytes);

    SparseMatrix A;
    A.row = (int *)malloc((nx * ny + 1) * sizeof(int)); // Allcate raw memory
    A.col = (int *)malloc(nnz_estimate * sizeof(int));
    A.value = (double *)malloc(nnz_estimate * sizeof(double));
    A.nnz = 0;

    // Flatten input arrays
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            Y_n[i * ny + j] = Y[i][j];
            u_flatten[i * ny + j] = u[i][j];
            v_flatten[i * ny + j] = v[i][j];
        }
    }

    // Allocate device memory
    double *d_Yn, *d_x, *d_b_flatten, *d_u, *d_v;
    double *d_values;
    int *d_column_indices, *d_row_offsets;

    CHECK_ERROR(cudaMalloc((void **)&d_Yn, unidimensional_size_bytes));
    CHECK_ERROR(cudaMalloc((void **)&d_x, unidimensional_size_bytes));
    CHECK_ERROR(cudaMalloc((void **)&d_b_flatten, unidimensional_size_bytes));
    CHECK_ERROR(cudaMalloc((void **)&d_u, unidimensional_size_bytes));
    CHECK_ERROR(cudaMalloc((void **)&d_v, unidimensional_size_bytes));
    CHECK_ERROR(cudaMalloc((void **)&d_values, nnz_estimate * sizeof(double)));
    CHECK_ERROR(cudaMalloc((void **)&d_column_indices, nnz_estimate * sizeof(int)));
    CHECK_ERROR(cudaMalloc((void **)&d_row_offsets, (nx * ny + 1) * sizeof(int)));

    // Copy input data to device
    CHECK_ERROR(cudaMemcpy(d_Yn, Y_n, unidimensional_size_bytes, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(d_u, u_flatten, unidimensional_size_bytes, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(d_v, v_flatten, unidimensional_size_bytes, cudaMemcpyHostToDevice));

    dim3 blockDim(10, 10);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y);
    auto end_init_solve = duration_cast<microseconds>(high_resolution_clock::now() - start_total_solve).count();
    printf("[SOLVE] Initialization took: %ld us\n", end_init_solve);
    auto start_fillMatrix = high_resolution_clock::now();

    // Fill A
    fillMatrixAKernel<<<gridDim, blockDim>>>(d_values, d_column_indices, d_row_offsets, dx, dy, D, dt, nx, ny);
    auto end_fillMatrix = duration_cast<microseconds>(high_resolution_clock::now() - start_fillMatrix).count();
    printf("[SOLVE] Fill Matrix A took: %ld us\n", end_fillMatrix);

    // Compute b
    auto start_fillb = high_resolution_clock::now();

    computeB<<<gridDim, blockDim>>>(d_b_flatten, d_Yn, d_u, d_v, dx, dy, nx, ny, dt);
    auto end_fillb = duration_cast<microseconds>(high_resolution_clock::now() - start_fillb).count();
    printf("[SOLVE] Fill b took: %ld us\n", end_fillb);
    cudaDeviceSynchronize();

    // Copy results back to the host
    CHECK_ERROR(cudaMemcpy(b_flatten, d_b_flatten, unidimensional_size_bytes, cudaMemcpyDeviceToHost));

    //  Jacobi Solver
    
    CHECK_ERROR(cudaMemcpy(A.row, d_row_offsets, (nx * ny + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_ERROR(cudaMemcpy(A.col, d_column_indices, nnz_estimate * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_ERROR(cudaMemcpy(A.value, d_values, nnz_estimate * sizeof(double), cudaMemcpyDeviceToHost));
    A.nnz = A.row[nx * ny];
    auto start_computex = high_resolution_clock::now();

    jacobiSolver(A, b_flatten, x, nx * ny, 1000, 1e-2);
    auto end_computex = duration_cast<microseconds>(high_resolution_clock::now() - start_computex).count();
    printf("[SOLVE] Fill x took: %ld us\n", end_computex);

    // Update Y
    for (int i = 1; i < nx - 1; i++)
    {
        for (int j = 1; j < ny - 1; j++)
        {
            Y[i][j] = x[i * ny + j];
        }
    }

    computeBoundaries(Y, nx, ny);

    // Free device memory
    cudaFree(d_Yn);
    cudaFree(d_x);
    cudaFree(d_b_flatten);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_values);
    cudaFree(d_column_indices);
    cudaFree(d_row_offsets);

    // Free host memory
    free(Y_n);
    free(x);
    free(b_flatten);
    free(A.row);
    free(A.col);
    free(A.value);

    auto end_total_solve = duration_cast<microseconds>(high_resolution_clock::now() - start_total_solve).count();
    printf("[SOLVE] Total time taken: %ld us\n", end_total_solve);
}
