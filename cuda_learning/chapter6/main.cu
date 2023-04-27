#include "kernel.h"
#include <stdio.h>
#define n_streams 4
int main()
{
    cudaStream_t *streams = (cudaStream_t *)malloc(n_streams * sizeof(cudaStream_t)); 
    for (int i = 0 ; i < n_streams; i++)  cudaStreamCreate(&streams[i]); 
    cudaEvent_t start, stop; 
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);
    dim3 block(1); 
    dim3 grid(1); 
    int num = 128;
    cudaEventRecord(start);
    for (int i = 0; i < n_streams; i++) 
    { 
        kernel_1<<<grid, block, 0, streams[i]>>>(num); 
        kernel_2<<<grid, block, 0, streams[i]>>>(num);
        kernel_3<<<grid, block, 0, streams[i]>>>(num); 
        kernel_4<<<grid, block, 0, streams[i]>>>(num); 
    }
    cudaEventRecord(stop);
    float elapsed_time =0.0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("%lf",elapsed_time);
}