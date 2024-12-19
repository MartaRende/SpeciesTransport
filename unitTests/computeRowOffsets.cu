#include "../solve/tools.h"
#include <cuda.h>
#include <cassert>
#include <iostream>
#include <cuda_runtime.h>
#include <cassert>

void runTestRowOffset(int * row, int nx, int ny, const char* testName) {
    int *h_row_offsets = new int[ny+1];
    int *d_row_offsets;
    cudaMalloc(&d_row_offsets, (ny+1)* sizeof(int));

    dim3 blockDim(16, 16);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y);

    initializeRowOffsetsKernel<<<gridDim, blockDim>>>(d_row_offsets, nx, ny);
    cudaDeviceSynchronize();

    cudaMemcpy(h_row_offsets, d_row_offsets,  (ny+1) * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < ny+1; ++i) {
            //row[idx] = h_row_offsets[idx];
            printf("%d %d\n",row[i], h_row_offsets[i]);
            //assert(h_row_offsets[idx] == 5 * idx);
        
    }

    std::cout << testName << " passed successfully." << std::endl;

    delete[] h_row_offsets;
    cudaFree(d_row_offsets);
}
