#include <iostream>
#include "tools.h"
__global__ void jacobiKernel(int *row, int *col, double *value, double *b, double *x, double *x_new, int nx, int ny, int nnz, int max_iterations, double tolerance)
{
    // 2D block and grid dimensions
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    // Check if thread is within bounds
    /* if (i < ny-1 && j < nx-1)
    { */
    if (i < ny && j < nx)
    {
        {

            for (int iter = 0; iter < max_iterations; ++iter)
            {
                int idx = i * nx + j; // 2D index flattened to 1D
                double sum = 0.0;
                double diag = 1.0;
                int row_start = row[idx];   // Starting index for this row in the sparse matrix
                int row_end = row[idx + 1]; // Ending index for this row

                // Calculate the sum and diagonal for Jacobi iteration
                for (int k = row_start; k < row_end; k++)
                {  /* if(x_new[col[k]]!=0.0) */
                  //  printf("ehoo %d %d %f\n",row_start,col[k], x_new[col[k]]);

                    if (col[k] == idx) // Diagonal element
                    {

                        diag = value[k];
                        /* if(value[k]==0.0){
                            printf("la %d %d\n",row_start,row_end);




                         }  */
                    }
                    else if(col[k]>0)
                    {
                        sum += value[k] * x_new[col[k]]; // Use x_new for the previous iteration
                                                         //  printf("sum %f %f %d %d %d \n", sum, x_new[col[k]],idx, row_start, row_end);
                        /*  if (x_new[col[k]] != 0.0)
                           printf("sum %f %f %d\n",sum,x_new[col[k]],col[k]);  */
                    }
                }

                // Calculate the new value for this element in the grid
                double new_value = (b[idx] - sum) / diag;
                const double epsilon = 1e-10; // Define a small threshold
               /*  if (fabs(new_value) > epsilon)
                {                printf("ahahah %f %f\n", b[idx] - sum, diag);

                    // printf("%f\n",b[idx]);
                     printf("val %f\n", new_value);
                } */

                // Update the new value for x_new

                // Check for convergence (based on the tolerance)
                if (fabs(new_value - x_new[idx]) < tolerance)
                {
                    x_new[idx] = new_value;

                    break;
                }
                else
                {
                    x_new[idx] = new_value;
                }

                // Synchronize threads before next iteration (not strictly necessary in this case)
                __syncthreads();
            }
        }
    }
}
__global__ void fillMatrixAKernel(double *values, int *column_indices, int *row_offsets,
                                  const double dx, const double dy, const double D,
                                  const double dt, const int nx, const int ny)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Indice colonna
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Indice riga

    int idx = i * nx + j;
    int c = 0;

    if (i < ny && j < nx)
    {

        int row_start = row_offsets[idx];
        // printf("%d\n",row_start);
        int count = 0;

        double diag_val = 1 + dt * D * (2 / (dx * dx) + 2 / (dy * dy));
        values[row_start] = diag_val;
        column_indices[row_start] = idx;
        count++;
        if (j > 0)
        {
            values[row_start + 1] = -dt * D / (dx * dx);
            column_indices[row_start + 1] = idx - 1;
        }
        if (j < nx - 1)
        {
            values[row_start + 2] = -dt * D / (dx * dx);
            column_indices[row_start + 2] = idx + 1;
            count++;
        }
        if (i < nx - 1)
        {
            values[row_start + 3] = -dt * D / (dy * dy);
            column_indices[row_start + 3] = idx + nx;
            count++;
        }
        if (i > 0)
        {
            values[row_start + 4] = -dt * D / (dy * dy);
            column_indices[row_start + 4] = idx - nx;
            count++;
        }
        //  printf("col %d %d %d %d %d\n", column_indices[row_start], column_indices[row_start + 1], column_indices[row_start + 2], column_indices[row_start + 3], column_indices[row_start + 4]);
    }
}

__global__ void computeB(double *b, double *Y_n, double *u, double *v,
                         const double dx, const double dy, const int nx, const int ny, const double dt)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = i * nx + j;
    if (i >= ny - 1 || j >= nx - 1)
        return;
    if (i == 0 && j == 0)
        return;

    /*   if (i == 0 || i == ny - 1 || j == 0 || j == nx - 1)
          return; */

    int right = i * nx + (j + 1);
    int left = i * nx + (j - 1);
    int top = (i - 1) * nx + j;
    int down = (i + 1) * nx + j;

    b[idx] = Y_n[idx];

    if (u[idx] < 0.0)
        b[idx] -= dt * (u[idx] * (Y_n[down] - Y_n[idx]) / dx);
    else
        b[idx] -= dt * (u[idx] * (Y_n[idx] - Y_n[top]) / dx);

    if (v[idx] < 0.0)
        b[idx] -= dt * (v[idx] * (Y_n[right] - Y_n[idx]) / dy);
    else
        b[idx] -= dt * (v[idx] * (Y_n[idx] - Y_n[left]) / dy);
 //   if (b[idx] != 0.0)

      //  printf(" b is %f\n", b[idx]);
}
__global__ void initializeRowOffsetsKernel(int *row_offsets, const int nx, const int ny)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    if (i >= ny || j >= nx)
        return; // Bounds check
    int idx = i * nx + j;
    // Row offsets for sparse matrix. Each row will have exactly 5 elements.
    // Row 0 starts at 0, Row 1 starts at 5, Row 2 starts at 10, etc.
    row_offsets[idx] = idx * 5;

    // Debug print (optional): To confirm correct row offsets
}
