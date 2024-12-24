/* #include <mpi.h>
#include "solve.h"
void jacobiSolverSparseMPI(SparseMatrix &A, double *b, double *x, int n, int max_iter, double tol)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_process = n / size;  // Number of rows each process will handle
    int remainder = n % size;         // Handle any remainder if n is not divisible by size

    int start_row = rank * rows_per_process + (rank < remainder ? rank : remainder);
    int end_row = start_row + rows_per_process + (rank < remainder ? 1 : 0);

    double *local_x = new double[n]; // Local copy of the x vector
    double *local_x_new = new double[n]; // Local copy of the new x vector
    double *local_b = new double[n]; // Local copy of the b vector

    // Scatter the vector b to all processes
    MPI_Scatter(b, n / size, MPI_DOUBLE, local_b, n / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Initialize local_x with the current solution (if needed)
    for (int i = start_row; i < end_row; ++i)
    {
        local_x[i] = x[i];
    }

    // Jacobi iteration loop
    for (int iter = 0; iter < max_iter; ++iter)
    {
        // Step 1: Apply the Jacobi update rule for local rows (sparse matrix)
        for (int i = start_row; i < end_row; ++i)
        {
            double sum = 0.0;
            int row_start = A.row[i];
            int row_end = A.row[i + 1];

            // Loop through non-zero elements in the row
            for (int j = row_start; j < row_end; ++j)
            {
                int col = A.col[j];
                if (i != col) // Skip diagonal elements
                {
                    sum += A.value[j] * local_x[col];
                }
            }

            // Jacobi update
            local_x_new[i] = (local_b[i] - sum) / A.value[row_start]; // Diagonal value is at the start of the row
        }

        // Step 2: Communicate boundary values (neighboring rows)
        if (rank > 0) // Send to the previous process
        {
            MPI_Send(&local_x_new[start_row], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&local_x_new[start_row - 1], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (rank < size - 1) // Receive from the next process
        {
            MPI_Recv(&local_x_new[end_row], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&local_x_new[end_row - 1], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
        }

        // Step 3: Check for convergence by gathering local differences
        double local_diff = 0.0;
        for (int i = start_row; i < end_row; ++i)
        {
            local_diff += abs(local_x_new[i] - local_x[i]);
        }

        // Reduce the local differences to find the global difference
        double global_diff = 0.0;
        MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Step 4: If the global difference is below the tolerance, break
        if (global_diff < tol)
        {
            break;
        }

        // Step 5: Update local x for the next iteration
        for (int i = start_row; i < end_row; ++i)
        {
            local_x[i] = local_x_new[i];
        }
    }

    // Gather the final solution from all processes to the root process
    MPI_Gather(local_x_new, n / size, MPI_DOUBLE, x, n / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Clean up
    delete[] local_x;
    delete[] local_x_new;
    delete[] local_b;
} */