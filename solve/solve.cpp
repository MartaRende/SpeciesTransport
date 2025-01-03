#include "solve.h"
#include <stdio.h>
#include <iostream>
#include <unordered_map>

#include <chrono>
using namespace std;
using namespace chrono;
#include <vector>


void solveSpeciesEquation(double **Y, double **u, double **v, const double dx, const double dy, double D, const int nx, const int ny, const double dt)
{  
    auto start_total_solve = high_resolution_clock::now();

     // == arrays needed ==
    double **Y_n = new double *[nx]; // Previous Y
    double *x = new double[nx * ny];
    SparseMatrix A;
    double *b_flatten = new double[nx * ny];
    
    // == array initialisation
    for (int i = 0; i < nx * ny; ++i)
    {
        x[i] = 0.0;
     
    }

    for (int i = 0; i < nx; i++)
    {
        Y_n[i] = new double[ny];

        for (int j = 0; j < ny; j++)
        {
            Y_n[i][j] = Y[i][j]; // save Y in Y_n to save previous results
        }
    }
   

    auto end_init_solve = duration_cast<microseconds>(high_resolution_clock::now() - start_total_solve).count();
    printf("[SOLVE] Initialization took: %ld us\n", end_init_solve);
    // == Compute part Ax = b ==
    //Fill A with right coeffs.
    auto start_fillMatrix = high_resolution_clock::now();
    fillMatrixA(A, dx, dy, D, dt, nx, ny);
    auto end_fillMatrix = duration_cast<microseconds>(high_resolution_clock::now() - start_fillMatrix).count();
    printf("[SOLVE] Fill Matrix A took: %ld us\n", end_fillMatrix);

    //== Compute b part == 

    auto start_fillb = high_resolution_clock::now();
    for (int i = 1; i < nx - 1; i++)
    {
        for (int j = 1; j < ny - 1; j++)
        {

            b_flatten[i * ny + j] = Y_n[i][j];

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

    // == Compute x with an iterative method ==

    auto start_computex = high_resolution_clock::now();
    jacobiSolver(A, b_flatten, x, nx * ny,1000,1e-20);
    auto end_computex = duration_cast<microseconds>(high_resolution_clock::now() - start_computex).count();
    printf("[SOLVE] Fill x took: %ld us\n", end_computex);

    // Update Y

    for (int i = 1; i < nx - 1; i++)
    {
        for (int j = 1; j < ny - 1; j++)
        {    

            Y[i][j] = x[i * ny + j];
        }
    }
    // == Compute boundaries ==  

    computeBoundaries(Y, nx, ny); 

    // == Free memory ==
    for (int i = 0; i < nx; ++i) {
    delete[] Y_n[i];
    }
    delete[] Y_n;

    delete[] x;
    delete[] b_flatten;

    auto end_total_solve = duration_cast<microseconds>(high_resolution_clock::now() - start_total_solve).count();
    printf("[SOLVE] Total time taken: %ld us\n", end_total_solve);
}
