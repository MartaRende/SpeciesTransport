#include <iostream>
#include "tools.h"

// == kernel to compute boundaries ==
__global__ void computeBoundariesKernel(double *Y, const int nx, const int ny)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * nx + j;
    if (i < ny && j < nx) // make sure to be in the domain
    {
        if (j == 0 || j == nx - 1 || i == 0 || i == ny - 1)
        {
            Y[idx] = 0.0; // set boundaries to 0 for simplicity
        }
    }
}
// == kernel to compute jacobi method ===
__global__ void jacobiKernel(int *row, int *col, double *value, double *b, double *x, double *x_new, int nx, int ny, int nnz, int max_iterations, double tolerance)
{
    // 2D block and grid dimensions
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
            int idx = i * nx + j; // 2D index flattened to 1D

    if (i < ny && j < nx && idx < nx  && idx <ny) // make sure to be in the domain
    {

        for (int iter = 0; iter < max_iterations; ++iter)
        {
            double sum = 0.0;
            double diag = 1.0;
            int row_start = row[idx];   // Starting index for this row in the sparse matrix
            int row_end = row[idx + 1]; // Ending index for this row

            // Calculate jacobi formula
            for (int k = row_start; k < row_end; k++)
            {
                if (col[k] == idx){
                    diag = value[k]; // save value of diagonals
                }
                else
                    sum += value[k] * x[col[k]]; // calculate sum part
            }
            // Calculate the new value for this element in the grid
            double new_value = (b[idx] - sum) / diag;
            x_new[idx] = new_value;

            // Check for convergence (based on the tolerance)
            if (fabs(new_value - x[idx]) < tolerance)
                break;

            x[idx] = x_new[idx]; // save the new value of x
        }
    }
}

// == kernel to fill A values and A col. index ==
__global__ void fillMatrixAKernel(double *values, int *column_indices, int *row_offsets,
                                  const double dx, const double dy, const double D,
                                  const double dt, const int nx, const int ny)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = i * nx + j;

    if (i < ny && j < nx) // make sure to be in the domain
    {

        int row_start = row_offsets[idx];                                 // offset is necessary to fill each row correctly , All rows will then be filled with 5 values
        values[row_start] = 1 + dt * D * (2 / (dx * dx) + 2 / (dy * dy)); // diagonal value
        column_indices[row_start] = idx;

        values[row_start + 1] = -dt * D / (dx * dx); // first neighbor value ,value before diag
        column_indices[row_start + 1] = idx - 1;

        values[row_start + 2] = -dt * D / (dx * dx); // second neighbor value, value after diag
        column_indices[row_start + 2] = idx + 1;

        values[row_start + 3] = -dt * D / (dy * dy);
        column_indices[row_start + 3] = ((i - 3) * ny + j); // third neighbor value,distance of 3 values form diag

        values[row_start + 4] = -dt * D / (dy * dy);
        column_indices[row_start + 4] = ((i + 3) * ny + j); // fourth neighbor value,distance of 3 values form diag
    }
}

// == kernel to computer advection part ==
__global__ void computeB(double *b, double *Y_n, double *u, double *v,
                         const double dx, const double dy, const int nx, const int ny, const double dt)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = i * nx + j;

    if (i < ny && j < nx ) // make sure to be in the domain
    {
        int right = i * nx + (j + 1);
        int left = i * nx + (j - 1);
        int top = (i - 1) * nx + j;
        int down = (i + 1) * nx + j;

        b[idx] = Y_n[idx];
        // this division in necessary to compute ∇Y operation
        if (u[idx] < 0.0)
            b[idx] -= dt * (u[idx] * (Y_n[down] - Y_n[idx]) / dx);
        else
            b[idx] -= dt * (u[idx] * (Y_n[idx] - Y_n[top]) / dx);

        if (v[idx] < 0.0)
            b[idx] -= dt * (v[idx] * (Y_n[right] - Y_n[idx]) / dy);
        else
            b[idx] -= dt * (v[idx] * (Y_n[idx] - Y_n[left]) / dy);
    }
}
// == kernel to initialise row offset of A  ==
__global__ void initializeRowOffsetsKernel(int *row_offsets, const int nx, const int ny)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < ny && j < nx) // make sure to be in the domain
    {
        int idx = i * nx + j;
        row_offsets[idx] = idx * 5; // offset is 0,5,10,... for semplicity
    }
}
