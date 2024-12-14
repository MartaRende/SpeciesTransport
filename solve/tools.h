#ifndef TOOLS_H
#define TOOLS_H

extern __global__ void jacobiKernel(int *row, int *col, double *value,
                                    double *b, double *x_old, double *x_new, int n);

__global__ void computeDifferenceAtomic(double *x_new, double *x_old, double *d_diff, int N);
__device__ void atomicAdd_double(double* address, double value);

#endif
