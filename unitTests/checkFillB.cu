#include <iostream>
#include <cassert>
#include <cuda.h>
#include <cmath>
#include "../solve/tools.h"
#include "unitTest.h"
#include <cmath>
void testFillb(int nx, int ny, double dt, double dx, double dy, double *u, double *v, double *Yn, double *b_expeted, double *b)
{

    double *d_u, *d_v, *d_Yn;
    double *d_b;

    cudaMalloc(&d_b, nx * ny * sizeof(double));
    cudaMalloc(&d_u, nx * ny * sizeof(double));
    cudaMalloc(&d_Yn, nx * ny * sizeof(double));
    cudaMalloc(&d_v, nx * ny * sizeof(double));
   
    cudaMemcpy(d_u, u, nx * ny * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, nx * ny * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Yn, Yn, nx * ny * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blockDim(8,8);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y);
    computeB<<<gridDim, blockDim>>>(d_b, d_Yn, d_u, d_v, dx, dy, nx, ny, dt);
    cudaMemcpy(b, d_b, nx * ny * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nx*ny; i++)
    {
        printf("hold %f\n", b_expeted[i]);
        printf("new %f\n", b[i]);

        //    assert(fabs(x_new[i] - x[i]) < 0.5);
    }
    cudaFree(d_b);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_Yn);
}