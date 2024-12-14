__global__ void jacobiKernel(int *row, int *col, double *value,
                             double *b, double *x_old, double *x_new,
                             int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Thread ID corresponds to row index
    if (i >= n) return; // Out of bounds

    double sum = b[i]; // Start with b[i]
    int row_start = row[i];
    int row_end = row[i + 1];
    double diag_value = 0.0; // Diagonal element

    // Iterate through the sparse row
    for (int k = row_start; k < row_end; ++k)
    {
        int j = col[k]; // Column index
        if (i == j)
            diag_value = value[k]; // Diagonal element
        else
            sum -= value[k] * x_old[j]; // Off-diagonal contributions
    }

    // Update x_new[i]
    if (diag_value != 0.0)
        x_new[i] = sum / diag_value;
}
__device__ void atomicAdd_double(double* address, double value)
{
    unsigned long long* address_as_ull = (unsigned long long*) address;
    unsigned long long old = *address_as_ull;
    unsigned long long assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(value + __longlong_as_double(assumed)));
    } while (assumed != old);
}


__global__ void computeDifferenceAtomic(double *x_new, double *x_old, double *d_diff, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    double diff_value = fabs(x_new[idx] - x_old[idx]);

    // Atomically add to the global diff variable
    atomicAdd_double(d_diff, diff_value);
}
