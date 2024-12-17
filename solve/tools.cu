#include <iostream>
#include "tools.h"

__global__ void jacobiKernel(int *row, int *col, double *value, double *b, double *x, double *x_new, int nx, int ny, int nnz)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = i * nx + j;
    if (i < ny && j < nx)
    {
        double sum = 0.0;
        double diag = 1.0;
        int row_start = row[i];
        int row_end = row[i + 1];

        //printf("Thread (%d, %d): row_start=%d, row_end=%d\n", i, j, row_start, row_end);

        x_new[idx] = b[idx];

        for (int k = row_start; k < row_end; k++)
        {
            //printf("k=%d, col[k]=%d, idx=%d, value[k]=%f, x[col[k]]=%f\n", k, col[k], idx, value[k], x[col[k]]);
            
            if (col[k] == idx) {
                //printf("Diagonal element found at k=%d\n", k);
                diag = value[k];
            }
            else {
                sum += value[k] * x[col[k]];
                //printf("Updated sum: %f\n", sum);
            }
        }
//printf("%f\n", diag);
        x_new[idx] = (x_new[idx] - sum) / diag;

       // printf("Final result for (%d, %d): x_new=%f, sum=%f, diag=%f\n", i, j, x_new[idx], sum, diag);
    }
}


// CUDA kernel for computing difference
__global__ void diffKernel( double *x,  double *x_new, double *diff, int nx,  int ny)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nx * ny)
    {
        diff[idx] = fabs(x_new[idx] - x[idx]);
    }
}
__global__ void fillMatrixAKernel(double *values, int *column_indices, int *row_offsets,
                                  const double dx, const double dy, const double D,
                                  const double dt, const int nx, const int ny)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= ny || j >= nx || i == 0 || i == ny - 1 || j == 0 || j == nx - 1)
        return;

    int idx = i * nx + j;
    int count = 0;
 if (i >= ny || j >= nx)
        return;


    int row_start = row_offsets[idx];
    // Diagonal
    values[row_start + count] = 1 + dt * D * (2 / (dx * dx) + 2 / (dy * dy));
    column_indices[row_start + count] = idx;
    count++;

    // Left neighbor
    values[row_start + count] = -dt * D / (dx * dx);
        column_indices[row_start + count] = idx - 1;
        count++;


    // Right neighbor
  
    values[row_start + count] = -dt * D / (dx * dx);
        column_indices[row_start + count] = idx + 1;
        count++;


    // Top neighbor
    values[row_start + count] = -dt * D / (dy * dy);
        column_indices[row_start + count] = idx - nx;
        count++;


    // Bottom neighbor
    values[row_start + count] = -dt * D / (dy * dy);
        column_indices[row_start + count] = idx + nx;
        count++;

}


__global__ void computeB(double *b, double *Y_n, double *u, double *v,
                         const double dx, const double dy, const int nx, const int ny, const double dt)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i == 0 || i == ny - 1 || j == 0 || j == nx - 1)
        return;

    int idx = i * nx + j;
    int right = i * nx + (j + 1);
    int left = i * nx + (j - 1);
    int top = (i - 1) * nx + j;
    int down = (i + 1) * nx + j;

    b[idx] = Y_n[idx];

    if (u[idx] < 0.0)
        b[idx] -= dt * (u[idx] * (Y_n[down] - Y_n[idx]) / dx);
    else
        b[idx] -= dt * (u[idx] * (Y_n[idx] - Y_n[top]) / dx);

    if (v[idx] < 0.0)
        b[idx] -= dt * (v[idx] * (Y_n[right] - Y_n[idx]) / dy);
    else
        b[idx] -= dt * (v[idx] * (Y_n[idx] - Y_n[left]) / dy);
       // printf("%f\n", b[idx]);
}
__global__ void initializeRowOffsetsKernel(int *row_offsets, const int nx, const int ny) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    if (i >= ny || j >= nx) return; // Bounds check
int idx = i*nx+j;
    // Row offsets for sparse matrix. Each row will have exactly 5 elements.
    // Row 0 starts at 0, Row 1 starts at 5, Row 2 starts at 10, etc.
    row_offsets[idx] = idx * 5;

    // Debug print (optional): To confirm correct row offsets
  
}


