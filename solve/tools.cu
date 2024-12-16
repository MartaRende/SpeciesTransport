#include <iostream>
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

        for (int k = row_start; k < row_end; ++k)
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
