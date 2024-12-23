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
    if(i<ny-1 && j< ny-1){    

        for (int iter = 0; iter < 1; ++iter)
        { int idx = i * nx + j;  // 2D index flattened to 1D
            double sum = 0.0;
            double diag = 1.0;
            int row_start = row[idx]; // Starting index for this row in the sparse matrix
            int row_end = row[idx + 1]; // Ending index for this row

            // Calculate the sum and diagonal for Jacobi iteration
            for (int k = row_start; k < row_end; k++)
            {               
                    //printf("%f %d\n",value[k],idx);

                if (col[k] == idx) // Diagonal element
                {    
                                
                    diag = value[k];
                   /* if(value[k]==0.0){
                       printf("la %d %d\n",row_start,row_end);
               
               
              

                    }  */


                }
                else 
                {
                    sum += 0.0; // Use x_new for the previous iteration
                  //  printf("sum %f %f %d %d %d \n", sum, x_new[col[k]],idx, row_start, row_end); 

   //   printf("sum %f\n",x_new[col[k]]); 
                }

            }

            // Calculate the new value for this element in the grid
            double new_value = (b[idx] - sum) / diag;
           
const double epsilon = 1e-10; // Define a small threshold
if (fabs(new_value) > epsilon) {
    // printf("%f\n",b[idx]);
   // printf("val %f\n", new_value);
}

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
            __syncthreads();
        }
    }
    
}__global__ void fillMatrixAKernel(double *values, int *column_indices, int *row_offsets,
                                   const double dx, const double dy, const double D,
                                   const double dt, const int nx, const int ny)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    if (i <= ny-1 && j <= nx-1) {
        int idx = i * nx + j;
        int row_start = row_offsets[idx];

        double diag_val = 1 + dt * D * (2 / (dx * dx) + 2 / (dy * dy));
        values[row_start] = diag_val;  
        column_indices[row_start] = idx;

            values[row_start + 1] = -dt * D / (dx * dx);
            column_indices[row_start + 1] = idx + 1;
        

            values[row_start + 2] = -dt * D / (dx * dx);
            column_indices[row_start + 2] = idx - 1;
        

            values[row_start + 3] = -dt * D / (dy * dy);
            column_indices[row_start + 3] = idx + 2; // Use -nx for upward neighbor
        

            values[row_start + 4] = -dt * D / (dy * dy);
            column_indices[row_start + 4] = idx - 2; // Use -2*nx for two rows up
        
    }
}



__global__ void computeB(double *b, double *Y_n, double *u, double *v,
                         const double dx, const double dy, const int nx, const int ny, const double dt)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = i * nx + j;
/*     if (i >= ny - 1 || j >= nx - 1)
        return; 
       if ( i == 0  &&  j == 0 )
        return;    */

    if (i == 0 || i == ny - 1 || j == 0 || j == nx - 1)
        return;
    if(idx >= nx*ny)
        return; 

   // int idx = i * nx + j;
    int right = i * nx + (j + 1);
    int left = i * nx + (j - 1);
    int top = (i - 1) * nx + j;
    int down = (i + 1) * nx + j;

    b[idx] = Y_n[idx];

    if (u[idx] < 0.0){
        b[idx] -= dt * (u[idx] * (Y_n[down] - Y_n[idx]) / dx);
    }
    else
        b[idx] -= dt * (u[idx] * (Y_n[idx] - Y_n[top]) / dx);

    if (v[idx] < 0.0)
        b[idx] -= dt * (v[idx] * (Y_n[right] - Y_n[idx]) / dy);
    else
        b[idx] -= dt * (v[idx] * (Y_n[idx] - Y_n[left]) / dy);
    if(b[idx]!=0.0)
    printf("b is %d %f\n", idx, b[idx]);


}
__global__ void initializeRowOffsetsKernel(int *row_offsets, const int nx, const int ny) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    int i = blockIdx.y * blockDim.y + threadIdx.y; 

int idx = i*nx+j;
    if (idx > ny *ny*5) return; // Controllo dei limiti

    row_offsets[idx] = idx * 5; // Supponendo che ci siano sempre 5 elementi non zero per riga

   /*  if (idx == ny - 1) {
        row_offsets[idx + 1] = row_offsets[idx] + 5; // Gestisci l'ultima riga
    } */
}

