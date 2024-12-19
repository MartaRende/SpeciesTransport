#include <iostream>
#include "tools.h"
__global__ void jacobiKernel(int *row, int *col, double *value, double *b, double *x, double *x_new, int nx, int ny, int nnz, int max_iterations, double tolerance)
{
    // 2D block and grid dimensions
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    // Check if thread is within bounds
    if (i < ny && j < nx)
    {
        int idx = i * nx + j;  // 2D index flattened to 1D

        for (int iter = 0; iter < max_iterations; ++iter)
        {
            double sum = 0.0;
            double diag = 1.0;
            int row_start = row[j]; // Starting index for this row in the sparse matrix
            int row_end = row[j + 1]; // Ending index for this row

            // Calculate the sum and diagonal for Jacobi iteration
            for (int k = row_start; k < row_end; k++)
            {               
                   // printf("%f %d\n",value[k],idx);

                if (col[k] == idx) // Diagonal element
                {    

                                
                    diag = value[k];
                   if(value[k]==0.0){
               /*          printf("%d %d\n",row_start,row_end);
               
               
                printf("%d\n",value[k]); */

                    } 


                }
                else // Off-diagonal elements
                {
                    sum += value[k] * x_new[col[k]]; // Use x_new for the previous iteration

                }

            }

            // Calculate the new value for this element in the grid
            double new_value = (b[idx] - sum) / diag;

            // Update the new value for x_new
           
            // Check for convergence (based on the tolerance)
            if (fabs(new_value - x_new[idx]) < tolerance)
            {
                    x_new[idx] = new_value;

                break;
            }else{
    x_new[idx] = new_value;
            }
                     


            // Synchronize threads before next iteration (not strictly necessary in this case)
           // __syncthreads();
        }
    }
}


__global__ void fillMatrixAKernel(double *values, int *column_indices, int *row_offsets,
                                   const double dx, const double dy, const double D,
                                   const double dt, const int nx, const int ny)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index

if (i == 0 || i == ny - 1 || j == 0 || j == nx - 1 )
    return;

    int idx = i * nx + j;  // Flattened index

    // Calculate the number of non-zero elements in the current row
    int num_non_zero_elements = 1;  // Diagonal element (always non-zero)
    if (j > 0) num_non_zero_elements++;  // Left neighbor
    if (j < nx - 1) num_non_zero_elements++; // Right neighbor
    if (i > 0) num_non_zero_elements++; // Top neighbor
    if (i < ny - 1) num_non_zero_elements++; // Bottom neighbor

    // Update row_offsets[idx] to track the start of this row in the sparse matrix
    int row_start = row_offsets[idx];
    
    // Debugging: Print the row start and num_non_zero_elements
   // printf("idx: %d, row_start: %d, num_non_zero_elements: %d\n", idx, row_start, num_non_zero_elements);

    // Store the non-zero elements in values and column_indices for the current row
    int count = 0; // Counter to keep track of the number of elements added to this row

    // Diagonal (current element)
    double diag_val = 1 + dt * D * (2 / (dx * dx) + 2 / (dy * dy));
    values[row_start + count] = diag_val;  // Store diagonal value
    column_indices[row_start + count] = idx; // Store column index for diagonal element
    count++; // Increment the count of non-zero elements

    // Debugging: Print the diagonal value and its position
    //printf("Diagonal value for idx %d: %f, at position %d\n", idx, diag_val, row_start + count - 1);

    // Left neighbor
    if (j > 0) {
        values[row_start + count] = -dt * D / (dx * dx);
        column_indices[row_start + count] = idx - 1; // Left neighbor
        count++;
    }

    // Right neighbor
    if (j < nx - 1) {
        values[row_start + count] = -dt * D / (dx * dx);
        column_indices[row_start + count] = idx + 1; // Right neighbor
        count++;
    }

    // Top neighbor
    if (i > 0) {
        values[row_start + count] = -dt * D / (dy * dy);
        column_indices[row_start + count] = idx - nx; // Top neighbor
        count++;
    }

    // Bottom neighbor
    if (i < ny - 1) {
        values[row_start + count] = -dt * D / (dy * dy);
        column_indices[row_start + count] = idx + nx; // Bottom neighbor
        count++;
    }



      //row_offsets[idx + 1] = row_offsets[idx] + num_non_zero_elements;
    
}



__global__ void computeB(double *b, double *Y_n, double *u, double *v,
                         const double dx, const double dy, const int nx, const int ny, const double dt)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i == 0 || i == ny - 1 || j == 0 || j == nx - 1)
        return;

    int idx = i * nx + j;
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
    //  printf("%f\n", Y_n[idx]);
}
__global__ void initializeRowOffsetsKernel(int *row_offsets, const int nx, const int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Row index

    if (i >= ny+ 1)
        return; // Bounds check

    int offset = 0;
    for (int row = 0; row <= i; row++) {
        int num_non_zero_elements;
        if (row == 0 || row == ny-1) {
            num_non_zero_elements = 3 ;
        }
        else if (row == 1 || row == ny - 2) {
            num_non_zero_elements = 4;
        }
        else {
            num_non_zero_elements = 5;
        }
        
        if (row == i) {
            row_offsets[i] = offset;
        }
        offset += 5;
    }
}

