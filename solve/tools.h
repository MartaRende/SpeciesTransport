#ifndef TOOLS_H
#define TOOLS_H

__global__ void jacobiKernel(int *row, int *col, double *value, double *b, double *x, double *x_new, int nx, int ny, int nnz, int max_iter,double tol);
__global__ void diffKernel(double *x, double *x_new, double *diff, int nx, int ny);

__global__ void fillMatrixAKernel(double *values, int *column_indices, int *row_offsets,
                                  const double dx, const double dy, const double D,
                                  const double dt, const int nx, const int ny);
__global__ void computeB(double *b, double *Y_n, double *u, double *v,
                         const double dx, const double dy, const int nx, const int ny, const double dt);
__global__ void initializeRowOffsetsKernel(int *row_offsets, const int nx, const int ny);
__global__ void computeBoundariesKernel(double *Y, const int nx, const int ny);

#endif
