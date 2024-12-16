#include <math.h>

#include "init.h"
#include <stdio.h>
#include <cuda.h>
using namespace std;
__device__ double sign(const double x1, const double y1, const double x2, const double y2, const double x3, const double y3)
{
    return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3);
}

__device__ bool isInsideTriangle(const double x1, const double y1, const double x2, const double y2, const double x3, const double y3, const double x, const double y)
{
    double d1, d2, d3;
    bool has_neg, has_pos;

    d1 = sign(x, y, x1, y1, x2, y2);
    d2 = sign(x, y, x2, y2, x3, y3);
    d3 = sign(x, y, x3, y3, x1, y1);

    has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

    return !(has_neg && has_pos);
}

__device__ bool isInsideCircle(const double x1, const double y1, const double radius, const double x, const double y)
{
    double d = sqrt((x1 - x) * (x1 - x) + (y1 - y) * (y1 - y));
    return d <= radius;
}

__global__ void initKernel(double *Y, double *u, double *v, int nx, int ny, double dx, double dy, double ycenter, double xcenter, double radius, double size, int s)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < ny && j < nx)
    {
        double x = j * dx - xcenter;
        double y = i * dy - ycenter;
        int idx = i * nx + j;
        if (isInsideCircle(0.0, 0.0, radius, x, y) && s == 0)
        { // White
            Y[idx] = 1.0;
        }
        else if ((isInsideTriangle(0.0, -radius, 0.0, radius, size + radius, radius, x, y) or
                  isInsideTriangle(0.0, -radius, size + radius, -radius, size + radius, radius, x, y) or
                  isInsideCircle(size + radius, 0.0, radius, x, y)) &&
                 s == 1)
        { // Green
            Y[idx] = 15.0;
        }
        else if ((isInsideTriangle(size / 2.0, -size / 2.0, -size / 2.0, size / 2.0, radius, size + radius, x, y) or
                  isInsideTriangle(size / 2.0, -size / 2.0, size + radius, radius, radius, size + radius, x, y) or
                  isInsideCircle(size / 2.0 + radius, size / 2.0 + radius, radius, x, y)) &&
                 s == 2)
        { // Pink
            Y[idx] = -5.0;
        }
        else if ((isInsideTriangle(radius, 0.0, -radius, 0.0, -radius, size + radius, x, y) or
                  isInsideTriangle(radius, 0.0, radius, size + radius, -radius, size + radius, x, y) or
                  isInsideCircle(0.0, size + radius, radius, x, y)) &&
                 s == 3)
        { // Purple
            Y[idx] = 10.0;
        }
        else if ((isInsideTriangle(-size / 2.0, -size / 2.0, size / 2.0, size / 2.0, -radius, size + radius, x, y) or
                  isInsideTriangle(-size / 2.0, -size / 2.0, -size - radius, radius, -radius, size + radius, x, y) or
                  isInsideCircle(-size / 2.0 - radius, size / 2.0 + radius, radius, x, y)) &&
                 s == 4)
        { // Blue
            Y[idx] = 5.0;
        }
        else if ((isInsideTriangle(0.0, -radius, 0.0, radius, -size - radius, radius, x, y) or
                  isInsideTriangle(0.0, -radius, -size - radius, -radius, -size - radius, radius, x, y) or
                  isInsideCircle(-size - radius, 0.0, radius, x, y)) &&
                 s == 5)
        { // Yellow
            Y[idx] = -10.0;
        }
        else
        {
            Y[idx] = 0.0;
        }
        // init of speeds
        if (s == 0)
        {
            u[idx] = sin(2.0 * M_PI * j * dy) * sin(M_PI * i * dx) * sin(M_PI * i * dx);
            v[idx] = -sin(2.0 * M_PI * i * dx) * sin(M_PI * j * dy) * sin(M_PI * j * dy);
        }
    }
}

// Initialization of the temperature inside the domain
void Initialization(double *Y, double *u, double *v, const int nx, const int ny, const double dx, const double dy, const int s, double * d_Y, double * d_u,double* d_v)
{

    // ISC LOGO
    size_t unidimensional_size_of_bytes = nx * ny * sizeof(double);
    // == Logo parameters ==
    double xcenter = 0.6;  // Logo position x
    double ycenter = 0.65; // Logo position y
    double radius = 0.05;  // Logo scale
    double size = sqrt(2) / 2.0 * radius / 0.5;



    dim3 blockDim(10, 10);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y);

    initKernel<<<gridDim, blockDim>>>(d_Y, d_u, d_v, nx, ny, dx, dy, ycenter, xcenter, radius, size, s);

    cudaDeviceSynchronize();

    // Copy results back to host


}
