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
    int nx = 50;
    int ny = 50;
    double Lx = 1.0;
    double Ly = 1.0;
    double dx = Lx/(nx-1);
    double dy =Ly/(ny-1);

    // == Temporal ==
    double tFinal = 2.0;
    double dt = 0.005;
    int nSteps = int(tFinal / dt);
    double time = 0.0;

    // array initialization
    double **Y = new double *[nx];
    double **u = new double *[nx];
    double **v = new double *[nx];
    for (int i = 0; i < nx; i++)
    {
        Y[i] = new double[ny];
        u[i] = new double[ny];
        v[i] = new double[ny];
    }
    Initialization(Y, u, v, nx, ny, dx, dy); // Initialize the temperature field inside the domain

    computeBoundaries(Y, nx, ny);
    auto end_init = duration_cast<nanoseconds>(high_resolution_clock::now() - start_total).count();
    printf("Initialization took: %ld ns\n", end_init);

    // == Output ==
    string outputName = "output/speciesTransport_";
    int count = 0;

    // == First output ==
    // Write data in VTK format
    writeDataVTK(outputName, Y, u, v, nx, ny, dx, dy, count++);

    // Loop over time
    auto end_write_first_file = duration_cast<nanoseconds>(high_resolution_clock::now() - start_total).count();
    printf("Writing first file took: %ld ns\n", end_write_first_file);

    for (int step = 1; step <= nSteps; step++)
    {
        auto start_eq = high_resolution_clock::now();
        // Solve the thermal equation (energy) to compute the temperature at the next time
        solveSpeciesEquation(Y, u, v, dx, dy, D, nx, ny, dt);
        auto end_eq = duration_cast<nanoseconds>(high_resolution_clock::now() - start_eq).count();
        printf("Compute species eq took: %ld ns\n", end_eq);
        fflush(stdout);
        // Write output every 100 iterations
        if (step % 100 == 0)
        {
            auto start_write = high_resolution_clock::now();
            writeDataVTK(outputName, Y, u, v, nx, ny, dx, dy, count++);
            auto end_write = duration_cast<nanoseconds>(high_resolution_clock::now() - start_write).count();
            printf("Write file %d took: %ld ns\n", count, end_write);
        }
    }

    for (int i = 0; i < nx; i++)
    {
        delete[] Y[i];
        delete[] u[i];
        delete[] v[i];
    }
    delete[] Y;
    delete[] u;
    delete[] v;

    auto end_total = duration_cast<nanoseconds>(high_resolution_clock::now() - start_total).count();
    printf("Total time taken: %ld ns\n", end_total);

    return 0;
}
