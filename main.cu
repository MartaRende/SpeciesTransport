#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>

#include "./common_includes.c"

#include "solve/solve.h"
#include "write/write.h"
#include "initialization/init.h"
#include <cuda.h>
#include <chrono>
using namespace std;
using namespace chrono;

int main()
{
    auto start_total = high_resolution_clock::now();
    // default parameters
    double D = 0.005; // possible values from 0.001 to 0.025
    int nx = 50; // in parallel 800
    int ny = 50; // in parallel 800
    double Lx = 1.0;
    double Ly = 1.0;
    double dx = Lx / (nx - 1); // in final version 0.0077
    double dy = Ly / (ny - 1); // in final version 0.0077

    // == Temporal ==
    double tFinal = 2.0;
    double dt = 0.0005;
    int nSteps = int(tFinal / dt);

    int nSpecies = 1; // Number of species

    int unidimensional_size = nx * ny; 
    int unidimensional_size_of_bytes = unidimensional_size * sizeof(double);
       size_t nnz_estimate = nx * ny * 5;
   // Array of pointers to 2D arrays for each species
    double **Y = (double **)malloc(nSpecies * sizeof(double *));

    // Allocate memory for each species' 2D array
    for (int s = 0; s < nSpecies; s++)
    {
        Y[s] = (double *)malloc(unidimensional_size_of_bytes);      
    }

    // Velocity fields
    double *u = (double *)malloc(unidimensional_size_of_bytes);
    double *v = (double *)malloc(unidimensional_size_of_bytes);
  //CUDA part

    double *d_Y;
    double *d_u;
    double *d_v;

    cudaMalloc((void **)&d_Y, unidimensional_size_of_bytes);
    cudaMalloc((void **)&d_u, unidimensional_size_of_bytes);
    cudaMalloc((void **)&d_v, unidimensional_size_of_bytes);

        // Allocate device memory
    double *d_Yn, *d_x;
    double *d_values,  *d_x_new, *d_b_flatten;
    int *d_column_indices, *d_row_offsets;
    CHECK_ERROR(cudaMalloc((void **)&d_Yn, unidimensional_size_of_bytes));
    CHECK_ERROR(cudaMalloc((void **)&d_x, unidimensional_size_of_bytes));
    CHECK_ERROR(cudaMalloc((void **)&d_x_new, unidimensional_size_of_bytes));
    CHECK_ERROR(cudaMalloc((void **)&d_b_flatten, unidimensional_size_of_bytes));
    CHECK_ERROR(cudaMalloc((void **)&d_values, nnz_estimate * sizeof(double)));
    CHECK_ERROR(cudaMalloc((void **)&d_column_indices, nnz_estimate * sizeof(int)));
    CHECK_ERROR(cudaMalloc((void **)&d_row_offsets, (nx * ny + 1) * sizeof(int)));


    // Initialize all species and velocity fields
    for (int s = 0; s < nSpecies; s++)
    {
        Initialization(Y[s],u,v,  nx, ny, dx, dy, s, d_Y, d_u, d_v);
        computeBoundaries(Y[s], nx, ny);
        cudaMemcpy(Y[s], d_Y, unidimensional_size_of_bytes, cudaMemcpyDeviceToHost);

    }
   

    auto end_init = duration_cast<microseconds>(high_resolution_clock::now() - start_total).count();
    printf("[MAIN] Initialization took: %ld us\n", end_init);

    // == Output ==
    string outputName = "output/speciesTransport_";
    int count = 0;
    
    //cudaMemcpy(u, d_u, unidimensional_size_of_bytes, cudaMemcpyDeviceToHost);
    //cudaMemcpy(v, d_v, unidimensional_size_of_bytes, cudaMemcpyDeviceToHost);
    // == First output ==
    writeDataVTK(outputName, Y, u, v, nx, ny, dx, dy, count++, nSpecies);

    auto end_write_first_file = duration_cast<microseconds>(high_resolution_clock::now() - start_total).count();
    printf("[MAIN] Writing first file took: %ld us\n", end_write_first_file);
    
    for (int step = 1; step <= nSteps; step++)
    {
        auto start_eq = high_resolution_clock::now();
        // Solve species equation
        for (int s = 0; s < nSpecies; s++)
        {
            solveSpeciesEquation(Y[s], dx, dy, D, nx, ny, dt, d_u,d_v,d_Yn,d_x,d_x_new,d_b_flatten,d_values, d_column_indices,d_row_offsets );
        }
        auto end_eq = duration_cast<microseconds>(high_resolution_clock::now() - start_eq).count();
        printf("[MAIN] Compute species eq took: %ld us\n", end_eq);
        
        // Write output every 100 iterations
        if (step % 100 == 0)
        {   
            auto start_write = high_resolution_clock::now();
            writeDataVTK(outputName, Y, u, v, nx, ny, dx, dy, count++, nSpecies);
            auto end_write = duration_cast<microseconds>(high_resolution_clock::now() - start_write).count();
            printf("[MAIN] Write file %d took: %ld us\n", count, end_write);
        }
    }

    // Free memory using free()
    for (int s = 0; s < nSpecies; s++)
    {
        free(Y[s]); // Free the pointer to the array of rows for each species
    }
    free(Y); // Free the pointer to the array of species

  
    free(u); // Free the pointer to the array of u rows
    free(v); // Free the pointer to the array of v rows
    
    
    cudaFree(d_Y);
    cudaFree(d_u);
    cudaFree(d_v);

     // Free device memory
    cudaFree(d_Yn);
    cudaFree(d_x);
    cudaFree(d_b_flatten);
 
    cudaFree(d_values);
    cudaFree(d_column_indices);
    cudaFree(d_row_offsets);
    auto end_total = duration_cast<microseconds>(high_resolution_clock::now() - start_total).count();
    printf("[MAIN] Total time taken: %ld us\n", end_total);

    return 0;
}
