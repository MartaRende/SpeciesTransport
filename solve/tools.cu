#include <iostream>

__global__ void jacobiKernel(int *row, int *col, double *value, double *b, double *x, double *x_new, int nx, int ny, int nnz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = i * ny + j;
    if (i < nx && j < ny)
    {
       
        double sum = 0.0;
        double diag = 1.0;
        int row_start = row[idx];
        int row_end = row[idx + 1];

        for (int k = row_start; k < row_end; ++k)
        {
            if (col[k] != idx)
                sum += value[k] * x[col[k]];
            else
                diag = value[k];
        }

        x_new[idx] = (b[idx] - sum) / diag;
   
    }
}

// CUDA kernel for computing difference
__global__ void diffKernel(double *x, double *x_new, double *diff, int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    //int idx = i*ny+j;
    if (i < nx && j < ny)
    {
        diff[i] = fabs(x_new[i] - x[i]);
    }
}