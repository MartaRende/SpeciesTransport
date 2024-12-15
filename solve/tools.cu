__global__ void jacobiKernel(int *row, int *col, double *value, double *b, double *x, double *x_new, int n, int nnz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double sum = 0.0;
        double diag = 1.0;
        int row_start = row[i];
        int row_end = row[i + 1];

        for (int k = row_start; k < row_end; ++k)
        {
            if (col[k] != i)
                sum += value[k] * x[col[k]];
            else
                diag = value[k];
        }

        x_new[i] = (b[i] - sum) / diag;
    }
}

// CUDA kernel for computing difference
__global__ void diffKernel(double *x, double *x_new, double *diff, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        diff[i] = fabs(x_new[i] - x[i]);
    }
}