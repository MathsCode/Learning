#include "../utils.h"
#include "./kernel.h"

int main(int argc,char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("%s using Device %d: %s\n", argv[0],dev, deviceProp.name);
    // set up data size 
    int size = 64; 
    int blocksize = 64; 
    if(argc > 1) blocksize = atoi(argv[1]); 
    if(argc > 2) size = atoi(argv[2]); 
    printf("Data size %d ", size); 
    // set up execution configuration 
    dim3 block (blocksize,1); 
    dim3 grid ((size+block.x-1)/block.x,1); 
    printf("Execution Configure (block %d grid %d)\n",block.x, grid.x); 
    // allocate gpu memory 
    float *d_C; 
    size_t nBytes = size * sizeof(float);
    cudaMalloc((float**)&d_C, nBytes); 
    // run a warmup kernel to remove overhead 
    float iStart,iElaps; 
    cudaDeviceSynchronize(); 
    iStart = cpuSecond(); 
    warmingup<<<grid, block>>> (d_C); 
    cudaDeviceSynchronize(); 
    iElaps = cpuSecond() - iStart; 
    printf("warmup <<< %4d %4d >>> elapsed %f sec \n",grid.x,block.x, iElaps ); 
    // run kernel 1 
    iStart = cpuSecond(); 
    mathKernel1<<<grid, block>>>(d_C); 
    cudaDeviceSynchronize(); 
    iElaps = cpuSecond() - iStart; 
    printf("mathKernel1 <<< %4d %4d >>> elapsed %f sec \n",grid.x,block.x,iElaps ); 
    // run kernel 2 
    iStart = cpuSecond(); 
    mathKernel2<<<grid, block>>>(d_C); 
    cudaDeviceSynchronize(); 
    iElaps = cpuSecond () - iStart; 
    printf("mathKernel2 <<< %4d %4d >>> elapsed %f sec \n",grid.x,block.x,iElaps ); 
    // run kernel 3 
    iStart = cpuSecond (); 
    mathKernel3<<<grid, block>>>(d_C); 
    cudaDeviceSynchronize(); 
    iElaps = cpuSecond () - iStart; 
    printf("mathKernel3 <<< %4d %4d >>> elapsed %f sec \n",grid.x,block.x,iElaps);
    // free gpu memory and reset divece 
    cudaFree(d_C);
    cudaDeviceReset(); 
    return EXIT_SUCCESS;   
}