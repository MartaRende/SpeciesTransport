#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <mpi.h>
#include "solve/solve.h"
#include "solve/tools.h"

#include "write/write.h"
#include "initialization/init.h"
#include <cuda.h>
#include <chrono>
#include "./common_includes.c"
using namespace std;
using namespace std::chrono;

int main(int argc, char *argv[])
{

    // == MPI Initialization ==
    MPI_Status status;
    int world_size, world_rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    auto start_total = high_resolution_clock::now();

    int *arrStart = new int[world_size];
    int *arrEnd = new int[world_size];
    int *splittedLengthes = new int[world_size];
    int *splittedSizes = new int[world_size];

    // default parameters
    double D = 0.005; // possible values from 0.001 to 0.025
    int nx = 50;      // in parallel 800
    int ny = 50;      // in parallel 800
    double Lx = 1.0;
    double Ly = 1.0;
    double dx = Lx / (nx - 1); // in final version 0.0077
    double dy = Ly / (ny - 1); // in final version 0.0077

    // == Temporal ==
    double tFinal = 2.0;
    double dt = 0.0005;
    int nSteps = int(tFinal / dt);

    int nSpecies = 1; // Number of species

    double **Y_splietted = (double **)malloc(nSpecies * sizeof(double *));
    double *u_splitted = new double[splittedLengthes[world_rank]];
    double *v_splitted = new double[splittedLengthes[world_rank]];

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

    // CUDA part
    double *d_Y, *d_u, *d_v;
    double *d_Yn, *d_x, *d_values, *d_x_new, *d_b_flatten;
    int *d_column_indices, *d_row_offsets;

    if (world_rank == 0)
    {
        cudaMalloc((void **)&d_Y, unidimensional_size_of_bytes);
        cudaMalloc((void **)&d_u, unidimensional_size_of_bytes);
        cudaMalloc((void **)&d_v, unidimensional_size_of_bytes);

        CHECK_ERROR(cudaMalloc((void **)&d_Yn, unidimensional_size_of_bytes));
        CHECK_ERROR(cudaMalloc((void **)&d_x, unidimensional_size_of_bytes));
        CHECK_ERROR(cudaMalloc((void **)&d_x_new, unidimensional_size_of_bytes));
        CHECK_ERROR(cudaMalloc((void **)&d_b_flatten, unidimensional_size_of_bytes));
        CHECK_ERROR(cudaMalloc((void **)&d_values, nnz_estimate * sizeof(double)));
        CHECK_ERROR(cudaMalloc((void **)&d_column_indices, nnz_estimate * sizeof(int)));
        CHECK_ERROR(cudaMalloc((void **)&d_row_offsets, (ny * nx) * sizeof(int)));
    }

    if (world_rank == 0)
    {
        int rest = unidimensional_size % world_size;
        int nbrOfElements = unidimensional_size / world_size;
        for (int i = 0; i < world_size; i++)
        {
            if (i < rest)
            {
                arrStart[i] = i * (nbrOfElements + 1);
                arrEnd[i] = (i + 1) * (nbrOfElements + 1);
                splittedLengthes[i] = (nbrOfElements + 1);
            }
            else
            {
                arrStart[i] = rest * (nbrOfElements + 1) + (i - rest) * nbrOfElements;
                arrEnd[i] = rest * (nbrOfElements + 1) + (i - rest + 1) * nbrOfElements;
                splittedLengthes[i] = nbrOfElements;
            }
            splittedSizes[i] = splittedLengthes[i] * sizeof(double);
        }
    }

    MPI_Bcast(arrStart, world_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(arrEnd, world_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(splittedLengthes, world_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(splittedSizes, world_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        for (int s = 0; s < nSpecies; s++)
        {
            Y_splietted[s] = (double *)malloc(splittedLengthes[world_rank]);
            Initialization(Y[s], u, v, nx, ny, dx, dy, s, d_Y, d_u, d_v);
            dim3 blockDim(16, 16);
            dim3 gridDim((nx + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y);
            computeBoundariesKernel<<<gridDim, blockDim>>>(d_Y, nx, ny);
            cudaMemcpy(Y[s], d_Y, unidimensional_size_of_bytes, cudaMemcpyDeviceToHost);
        }
    }

    // == Output ==
    string outputName = "output/speciesTransport_";
    int count = 0;

    if (world_rank == 0)
    {
        cudaMemcpy(u, d_u, unidimensional_size_of_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(v, d_v, unidimensional_size_of_bytes, cudaMemcpyDeviceToHost);
        for (int i = 1; i < world_size; i++)
        {
            for (int s = 0; s < nSpecies; s++)
            {
                MPI_Send(Y[s] + arrStart[i], splittedLengthes[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            }
            MPI_Send(u + arrStart[i], splittedLengthes[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            MPI_Send(v + arrStart[i], splittedLengthes[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
        Y_splietted = Y;
        u_splitted = u;
        v_splitted = v;
    }
    else
    {
        for (int s = 0; s < nSpecies; s++)
        {
            MPI_Recv(Y_splietted[s], splittedLengthes[world_rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        }
        MPI_Recv(u_splitted, splittedLengthes[world_rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(v_splitted, splittedLengthes[world_rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
    }

    string WriteU = getString(u_splitted, splittedLengthes[world_rank], world_rank);
    string WriteV = getString(v_splitted, splittedLengthes[world_rank], world_rank);

    string *WriteY = new string[nSpecies];
    /* for (int s = 0; s < nSpecies; s++)
     {
          CHECK_ERROR(cudaMemcpy(Y[s], d_Y, unidimensional_size_of_bytes, cudaMemcpyDeviceToHost));
     } */
    for (int s = 0; s < nSpecies; s++)
    {
        WriteY[s] = getString(Y_splietted[s], splittedLengthes[world_rank], world_rank);

    }
    // Launching write funtion with each part of the data to write
    writeDataVTK(outputName, WriteY, WriteU, WriteV, nx, ny, dx, dy, count++, world_rank, world_size, nSpecies);
    auto end_init = high_resolution_clock::now();

    auto initDuration = chrono::duration_cast<chrono::microseconds>(end_init - start_total).count(); // Calculate init duration
    printf("[MAIN] Initialization took: %ld us\n", initDuration);
    auto start_loop = high_resolution_clock::now();

    for (int step = 1; step <= nSteps; step++)
    {
        double max = 0;
        double total_length = 0;
        // Solve species equation
        if (world_rank == 0)
        {
            for (int s = 0; s < nSpecies; s++)
            {
                if(step==1){
                CHECK_ERROR(cudaMemcpy(d_Yn, Y[s], unidimensional_size_of_bytes, cudaMemcpyHostToDevice))

                }
                solveSpeciesEquation(Y[s], dx, dy, D, nx, ny, dt, d_u, d_v, d_Yn, d_x, d_x_new, d_b_flatten, d_values, d_column_indices, d_row_offsets);
          if (step % 100 == 0)
                {
                    CHECK_ERROR(cudaMemcpy(Y[s], d_x_new, unidimensional_size_of_bytes, cudaMemcpyDeviceToHost));
                }  
  
            }
            for (int i = 1; i < world_size; i++)
            {
                // Sending results part to each core
                for (int s = 0; s < nSpecies; s++)
                {
                    MPI_Send(Y + arrStart[i], splittedLengthes[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                }
            }
            Y_splietted = Y;
        }
        else
        {
            for (int s = 0; s < nSpecies; s++)
            {
                MPI_Recv(Y_splietted, splittedLengthes[world_rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
            }
        }

        if (step % 100 == 0)
        {
            auto start_write = high_resolution_clock::now();

            string *WriteY = new string[nSpecies];

            for (int s = 0; s < nSpecies; s++)
            {
                WriteY[s] = getString(Y_splietted[s], splittedLengthes[world_rank], world_rank);
            }

            writeDataVTK(outputName, WriteY, WriteU, WriteV, nx, ny, dx, dy, count++, world_rank, world_size, nSpecies);
            auto end_write = chrono::duration_cast<microseconds>(high_resolution_clock::now() - start_write).count();
            if (world_rank == 0)
                printf("[MAIN] Write file %d took: %ld us\n", count, end_write);
        }
    }
    auto end_loop = high_resolution_clock::now();
    auto loopDuration = duration_cast<microseconds>(end_loop - start_loop).count(); // Calculate loop duration
    printf("[MAIN] Loop took : %ld us\n", loopDuration);

    // Free memory using free()
    for (int s = 0; s < nSpecies; s++)
    {
        free(Y[s]); // Free the pointer to the array of rows for each species
    }
    free(Y); // Free the pointer to the array of species

    free(u); // Free the pointer to the array of u rows
    free(v); // Free the pointer to the array of v rows
    if (world_rank == 0)
    {
        // Free memory
        cudaFree(d_Yn);
        cudaFree(d_x);
        cudaFree(d_b_flatten);

        cudaFree(d_values);
        cudaFree(d_column_indices);
        cudaFree(d_row_offsets);
        cudaFree(d_Y);
        cudaFree(d_u);
        cudaFree(d_v);
    }

    auto end_total = high_resolution_clock::now();
    auto totalDuration = duration_cast<microseconds>(end_total - start_total).count(); // Calculate total duration

    printf("[MAIN] Total time taken: %ld us\n", totalDuration);

    MPI_Finalize();
    return 0;
}
