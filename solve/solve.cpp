#include "solve.h"
#include <stdio.h>
#include <iostream>

// #include "tools.h"

#include <chrono>
using namespace std;
using namespace chrono;
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
    auto start_total_solve = high_resolution_clock::now();
    double **Y_n = new double *[nx]; // Previous Y
    double *x = new double[nx * ny];
    double **A = new double *[nx * ny];
    double **A_inv = new double *[nx * nx];
    double *b_flatten = new double[nx * ny];

    for (int i = 0; i < nx * ny; ++i)
    {
        A[i] = new double[nx * ny]; // Allocate memory for each row
    }

    for (int i = 0; i < nx; i++)
    {
        Y_n[i] = new double[ny];

        for (int j = 0; j < ny; j++)
        {
            Y_n[i][j] = Y[i][j];
        }
    }
    for (int i = 0; i < nx * ny; i++)
    {
        A_inv[i] = new double[ny * ny];
        x[i] = 0.0;
    }
   

    auto end_init_solve = duration_cast<microseconds>(high_resolution_clock::now() - start_total_solve).count();
    printf("[SOLVE] Initialization took: %ld us\n", end_init_solve);
    auto start_fillMatrix = high_resolution_clock::now();
    fillMatrixA(A, dx, dy, D, dt, nx, ny);
    auto end_fillMatrix = duration_cast<microseconds>(high_resolution_clock::now() - start_fillMatrix).count();
    printf("[SOLVE] Fill Matrix A took: %ld us\n", end_fillMatrix);
    auto start_fillb = high_resolution_clock::now();
    for (int i = 1; i < nx - 1; i++)
    {
        for (int j = 1; j < ny - 1; j++)
        {

            b_flatten[i * ny + j] = Y_n[i][j];
            //+ dt*D  *( (Y_n[i+1][j] - 2.0 *Y_n[i][j] + Y_n[i-1][j])/dx/dx + (Y_n[i][j+1] - 2.0 *Y_n[i][j] + Y_n[i][j-1])/dy/dy );

            if (u[i][j] < 0.0)
            {
                b_flatten[i * ny + j] -= dt * (u[i][j] * (Y_n[i + 1][j] - Y_n[i][j]) / dx);
            }
            else
            {
                b_flatten[i * ny + j] -= dt * (u[i][j] * (Y_n[i][j] - Y_n[i - 1][j]) / dx);
            }

            if (v[i][j] < 0.0)
            {
                b_flatten[i * ny + j] -= dt * (v[i][j] * (Y_n[i][j + 1] - Y_n[i][j]) / dy);
            }
            else
            {
                b_flatten[i * ny + j] -= dt * (v[i][j] * (Y_n[i][j] - Y_n[i][j - 1]) / dy);
            }
        }
    }
    auto end_fillb = duration_cast<microseconds>(high_resolution_clock::now() - start_fillb).count();
    printf("[SOLVE] Fill b took: %ld us\n", end_fillb);
    auto start_computex = high_resolution_clock::now();
    conjugateGradient(A, b_flatten, x, nx * ny);
    auto end_computex = duration_cast<microseconds>(high_resolution_clock::now() - start_computex).count();
    printf("[SOLVE] Fill x took: %ld us\n", end_computex);
    for (int i = 1; i < nx - 1; i++)
    {
        for (int j = 1; j < ny - 1; j++)
        {

            Y[i][j] = x[i * ny + j];
        }
    }

    computeBoundaries(Y, nx, ny);
    for(int i = 0; i< nx*nx; i++){
        delete[] A_inv[i];
        delete[] A[i];
    }
    for (int i = 0; i < nx; ++i)
    {
        delete[] Y_n[i];
       
    }
    delete[] Y_n;
    delete[] x;
    delete[] b_flatten;
    delete[] A;
    delete[] A_inv;
    auto end_total_solve = duration_cast<microseconds>(high_resolution_clock::now() - start_total_solve).count();
    printf("[SOLVE] Total time taken: %ld us\n", end_total_solve);
}

void fillMatrixA(double **A, const double dx, const double dy, const double D, const double dt, const int nx, const int ny)
{
    // Loop through all grid points
    for (int i = 0; i < nx * ny; ++i)
    {
        for (int j = 0; j < nx * ny; ++j)
        {
            A[i][j] = 0.0; // Initialize with zero
        }
    }

    // Populate internal nodes
    for (int i = 1; i < nx - 1; ++i)
    {
        for (int j = 1; j < ny - 1; ++j)
        {
            int idx = i * ny + j;

            // Diagonal (central node)
            A[idx][idx] = 1 + dt * D * (2 / (dx * dx) + 2 / (dy * dy));

            // Left neighbor
            A[idx][idx - 1] = -dt * D / (dx * dx);

            // Right neighbor
            A[idx][idx + 1] = -dt * D / (dx * dx);

            // Top neighbor
            A[idx][(i - 1) * ny + j] = -dt * D / (dy * dy);

            // Bottom neighbor
            A[idx][(i + 1) * ny + j] = -dt * D / (dy * dy);
        }
    }

    // Handle boundary conditions (Dirichlet: Y = 0)
    for (int i = 0; i < nx; ++i)
    {
        // Bottom boundary
        A[i * ny][i * ny] = 1.0;

        // Top boundary
        A[i * ny + (ny - 1)][i * ny + (ny - 1)] = 1.0;
    }
    for (int j = 0; j < ny; ++j)
    {
        // Left boundary
        A[j][j] = 1.0;

        // Right boundary
        A[(nx - 1) * ny + j][(nx - 1) * ny + j] = 1.0;
    }
}
