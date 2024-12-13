#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>

#include "solve/solve.h"
#include "write/write.h"
#include "initialization/init.h"

#include <chrono>
using namespace std;
using namespace chrono;

int main()
{
    auto start_total = high_resolution_clock::now();

    // default parameters
    double D = 0.005; // possible values from 0.001 to 0.025
    int nx = 300; // in parallel 800
    int ny = 300; // in parallel 800
    double Lx = 1.0;
    double Ly = 1.0;
    double dx = Lx / (nx - 1); // in final version 0.0077
    double dy = Ly / (ny - 1); // in final version 0.0077

    // == Temporal ==
    double tFinal = 2.0;
    double dt = 0.0005;
    int nSteps = int(tFinal / dt);

    int nSpecies = 1; // Number of species

    // Array of pointers to 2D arrays for each species
    double ***Y = (double ***)malloc(nSpecies * sizeof(double **));

    // Allocate memory for each species' 2D array
    for (int s = 0; s < nSpecies; s++)
    {
        Y[s] = (double **)malloc(nx * sizeof(double *));
        for (int i = 0; i < nx; i++)
        {
            Y[s][i] = (double *)malloc(ny * sizeof(double));
        }
    }

    // Velocity fields
    double **u = (double **)malloc(nx * sizeof(double *));
    double **v = (double **)malloc(nx * sizeof(double *));
    for (int i = 0; i < nx; i++)
    {
        u[i] = (double *)malloc(ny * sizeof(double));
        v[i] = (double *)malloc(ny * sizeof(double));
    }

    // Initialize all species and velocity fields
    for (int s = 0; s < nSpecies; s++)
    {
        Initialization(Y[s], u, v, nx, ny, dx, dy, s);
        computeBoundaries(Y[s], nx, ny);
    }

    auto end_init = duration_cast<microseconds>(high_resolution_clock::now() - start_total).count();
    printf("[MAIN] Initialization took: %ld us\n", end_init);

    // == Output ==
    string outputName = "output/speciesTransport_";
    int count = 0;

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
            solveSpeciesEquation(Y[s], u, v, dx, dy, D, nx, ny, dt);
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
        for (int i = 0; i < nx; i++)
        {
            free(Y[s][i]); // Free each row of Y[s]
        }
        free(Y[s]); // Free the pointer to the array of rows for each species
    }
    free(Y); // Free the pointer to the array of species

    for (int i = 0; i < nx; i++)
    {
        free(u[i]); // Free each row of u
        free(v[i]); // Free each row of v
    }
    free(u); // Free the pointer to the array of u rows
    free(v); // Free the pointer to the array of v rows

    auto end_total = duration_cast<microseconds>(high_resolution_clock::now() - start_total).count();
    printf("[MAIN] Total time taken: %ld us\n", end_total);

    return 0;
}
