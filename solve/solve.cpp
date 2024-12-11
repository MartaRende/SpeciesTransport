#include "solve.h"
#include <stdio.h>
#include <iostream>

#include <Eigen/Dense>

using namespace Eigen;
// #include "tools.h"

using namespace std;

void computeBoundaries(double **Y, const int nx, const int ny)
{
    for (int i = 0; i < nx; i++)
    {
        Y[i][ny - 1] = 0.0;
        Y[i][0] = 0.0;
    }

    for (int j = 0; j < ny; j++)
    {
        Y[0][j] = 0.0;
        Y[nx - 1][j] = 0.0;
    }
}

void solveSpeciesEquation(double **Y, double **u, double **v, const double dx, const double dy, double D, const int nx, const int ny, const double dt)
{
    double **Y_n = new double *[nx]; // Previous Y
    double *x = new double[nx * ny];
    printf("here0");
    double **A = new double *[nx];

    double *b_flatten = new double[nx * ny];
    for (int i = 0; i < nx; i++)
    {
        Y_n[i] = new double[ny];
        A[i] = new double[ny];
        for (int j = 0; j < ny; j++)
        {
            Y_n[i][j] = Y[i][j];
            A[i][j] = Y[i][j];
        }
    }
    printf("here1");
    fflush(stdout);
    double **A_inv = new double *[nx];
    for (int i = 0; i < nx; i++)
    {
        A_inv[i] = new double[ny];
    }

    printf("here2");

    fflush(stdout);
    for (int i = 1; i < nx - 1; i++)
    {
        for (int j = 1; j < ny - 1; j++)
        {
            b_flatten[i * ny + j] = 0.0;

            // Y[i][j] = Y_n[i][j] + dt*D  *( (Y_n[i+1][j] - 2.0 *Y_n[i][j] + Y_n[i-1][j])/dx/dx + (Y_n[i][j+1] - 2.0 *Y_n[i][j] + Y_n[i][j-1])/dy/dy );

            if (u[i][j] < 0.0)
            {
                b_flatten[i * ny + j] = (u[i][j] * (Y_n[i + 1][j] - Y_n[i][j]) / dx);
            }
            else
            {
                b_flatten[i * ny + j] = (u[i][j] * (Y_n[i][j] - Y_n[i - 1][j]) / dx);
            }

            if (v[i][j] < 0.0)
            {
                b_flatten[i * ny + j] += (v[i][j] * (Y_n[i][j + 1] - Y_n[i][j]) / dy);
            }
            else
            {
                b_flatten[i * ny + j] += (v[i][j] * (Y_n[i][j] - Y_n[i][j - 1]) / dy);
            }
        }
    }
    if (determinant(A, nx) == 0)
    {
     MatrixXd MatA(nx, ny);
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                MatA(i, j) = A[i][j];
            }
        }
     MatrixXd A_pseudo_inv = computePseudoinverse(MatA);
      for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                A_inv[i][j]=A_pseudo_inv(i,j);
            }
        }

    }
    else
    {
        A_inv = invertMatrix(A, nx);
        
    }
    for (int i = 1; i < nx - 1; i++)
        {
            for (int j = 1; j < ny - 1; j++)
            {
                x[i * ny + j] = 0.0;
                for (int k = 1; k < ny - 1; k++)
                {
                    x[i * ny + j] += A_inv[i][k] * b_flatten[k];
                    if(x[i * ny + j]!= 0.0){
                    printf("%f", x[i * ny + j]);

                    }
                }
            }
        }

    printf("here3");
    for (int i = 1; i < nx - 1; i++)
    {
        for (int j = 1; j < ny - 1; j++)
        {
            int idx = i * ny + j;

            int idx_right = i * ny + (j + 1); // x[i][j+1]
            int idx_left = i * ny + (j - 1);  // x[i][j-1]
            int idx_up = (i - 1) * ny + j;    // x[i-1][j]
            int idx_down = (i + 1) * ny + j;  // x[i+1][j]

            Y[i][j] = Y_n[i][j] + dt * (-b_flatten[idx] + D * ((x[idx_down] - 2.0 * x[idx] + x[idx_up]) / (dx * dx) + (x[idx_right] - 2.0 * x[idx] + x[idx_left]) / (dy * dy)));
        }
    }
    printf("here4");

    computeBoundaries(Y, nx, ny);
    for (int i = 0; i < nx; ++i)
    {
        delete[] Y_n[i];
        delete[] A_inv[i];
    }
    delete[] Y_n;
    delete[] x;
    delete[] b_flatten;
    delete[] A_inv;
}