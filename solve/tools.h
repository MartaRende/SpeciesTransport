#ifndef TOOLS_H
#define TOOLS_H

__global__ void jacobiKernel(int *row, int *col, double *value, double *b, double *x, double *x_new, int nx, int ny, int nnz);
__global__ void diffKernel(double *x, double *x_new, double *diff, int n);

#endif
