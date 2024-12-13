#include <cmath>
#include <iostream>
#include <cuda.h>

using namespace std;


__global__ void jacobiSolverKernel(double* A, double* b, double* x, int size, int maxIterations, double tolerance, double* maxDiff) {
    extern __shared__ double shared_x_new[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i<size){
        shared_x_new[i]=x[i];
    }
    bool converged = false; 

    for (int iter = 0; iter < maxIterations; iter++) {
       if(i<size){
            double sum = 0.0;
            for (int j = 0; j < size; j++) {
                if (j != i) {
                    sum += A[i* size + j] * x[j];
                }
            }
            shared_x_new[i] = (b[i] - sum) / A[i * size + i];
        }
       /* if (i < size) {
            double diff = fabs(shared_x_new[i] - x[i]);
            atomicMax(&maxDiff, diff); 
        }

        if (maxDiff < tolerance) {
            converged=true;
            break;
        }*/
    }
    if (i < size && !converged) {
        x[i] = shared_x_new[i];
    }

   

}
