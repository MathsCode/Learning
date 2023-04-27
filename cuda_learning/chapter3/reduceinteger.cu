#include "../utils.cuh"
#include "fun.h"
int main(int argc, char **argv) 
{ 
    // set up device 
    int dev = 0; 
    if(argc > 0)
    {
        dev = atoi(argv[0]);
    }
    GetDeviceBasicInfo(dev);
    
    bool bResult = false; 
    // initialization 
    int size = 1<<24; // total number of elements to reduce 
    printf(" with array size %d ", size); 
    // execution configuration 
    int blocksize = 512; // initial block size 
    if(argc > 1) 
    { 
        blocksize = atoi(argv[1]); // block size from command line argument 
    } 
    dim3 block (blocksize,1); 
    dim3 grid ((size+block.x-1)/block.x,1); 
    printf("grid %d block %d\n",grid.x, block.x); 
    // allocate host memory 
    size_t bytes = size * sizeof(int); 
    int *h_idata = (int *) malloc(bytes); 
    int *h_odata = (int *) malloc(grid.x*sizeof(int)); //store the sum result of every block 
    int *tmp = (int *) malloc(bytes); 
    // initialize the array 
    fill_data(h_idata,size,0,255);
    memcpy (tmp, h_idata, bytes); 
    double iStart,iElaps; 
    // allocate device memory 
    int *d_idata = NULL; 
    int *d_odata = NULL; 
    cudaMalloc((void **) &d_idata, bytes); 
    cudaMalloc((void **) &d_odata, grid.x*sizeof(int)); 
    // cpu reduction 
    iStart = cpuSecond (); 
    int cpu_sum = recursiveReduce(tmp, size); 
    iElaps = cpuSecond() - iStart; 
    printf("cpu reduce elapsed %lf ms cpu_sum: %d\n",iElaps,cpu_sum); 


    // warmup kernel 1: reduceNeighbored 
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    int gpu_sum = 0; 
    cudaDeviceSynchronize(); 

    iStart = cpuSecond(); 
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size); 
    cudaDeviceSynchronize(); 
    iElaps = cpuSecond() - iStart; 
    
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost); 
    gpu_sum = 0; 
    for (int i=0; i<grid.x; i++) gpu_sum += h_odata[i]; 
    printf("gpu Warmup elapsed %lf ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps,gpu_sum,grid.x,block.x);


    // kernel 1: reduceNeighbored 
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice); 
    cudaDeviceSynchronize(); 
    iStart = cpuSecond (); 
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size); 
    cudaDeviceSynchronize(); 
    iElaps = cpuSecond () - iStart; 
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost); 
    gpu_sum = 0; 
    for (int i=0; i<grid.x; i++) gpu_sum += h_odata[i]; 
    printf("gpu Neighbored elapsed %lf ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps,gpu_sum,grid.x,block.x); 
    cudaDeviceSynchronize(); 


    // kernel 2: reduceNeighbored with less divergence 
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize(); 
    iStart = cpuSecond(); 
    reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, size); 
    cudaDeviceSynchronize(); 
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost); 
    gpu_sum = 0;
    for (int i=0; i<grid.x; i++) gpu_sum += h_odata[i]; 
    printf("gpu Neighbored2 elapsed %lf ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps,gpu_sum,grid.x,block.x);

    // kernel3: reduceUnrolling2

    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice); 
    cudaDeviceSynchronize(); 
    iStart = cpuSecond(); 
    reduceUnrolling2 <<< grid.x/2, block >>> (d_idata, d_odata, size); 
    cudaDeviceSynchronize(); 
    iElaps = cpuSecond() - iStart; 
    cudaMemcpy(h_odata, d_odata, grid.x/2*sizeof(int), cudaMemcpyDeviceToHost); 
    gpu_sum = 0; 
    for (int i = 0; i < grid.x / 2; i++) gpu_sum += h_odata[i]; 
    printf("gpu Unrolling2 elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps,gpu_sum,grid.x/2,block.x);


    // kernel4:reduceUnrolling8
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice); 
    cudaDeviceSynchronize(); 
    iStart = cpuSecond(); 
    reduceUnrollWarps8 <<< grid.x/8, block >>> (d_idata, d_odata, size); 
    cudaDeviceSynchronize(); 
    iElaps = cpuSecond() - iStart; 
    cudaMemcpy(h_odata, d_odata, grid.x/8*sizeof(int), cudaMemcpyDeviceToHost); 
    gpu_sum = 0; 
    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i]; 
    printf("gpu UnrollingWarps8 elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps,gpu_sum,grid.x/8,block.x);

    // kernel5:reduceUnrolling8v2
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice); 
    cudaDeviceSynchronize(); 
    iStart = cpuSecond(); 
    reduceUnrollWarps8 <<< grid.x/8, block >>> (d_idata, d_odata, size); 
    cudaDeviceSynchronize(); 
    iElaps = cpuSecond() - iStart; 
    cudaMemcpy(h_odata, d_odata, grid.x/8*sizeof(int), cudaMemcpyDeviceToHost); 
    gpu_sum = 0; 
    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i]; 
    printf("gpu UnrollingWarps8v2 elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps,gpu_sum,grid.x/8,block.x);

    // kernel6:Template reduceUnrolling8v2
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice); 
    cudaDeviceSynchronize(); 
    iStart = cpuSecond(); 
    switch (blocksize)
    {
        case 1024:
            reduceCompleteUnroll<1024><<< grid.x/8, block >>> (d_idata, d_odata, size); 
            break;
        case 512:
            reduceCompleteUnroll<512><<< grid.x/8, block >>> (d_idata, d_odata, size); 
            break;
        case 256:
            reduceCompleteUnroll<256><<< grid.x/8, block >>> (d_idata, d_odata, size); 
            break;
        case 128:
            reduceCompleteUnroll<128><<< grid.x/8, block >>> (d_idata, d_odata, size); 
            break;
        case 64:
            reduceCompleteUnroll<64><<< grid.x/8, block >>> (d_idata, d_odata, size); 
            break;
        default:
            break;
    }
    cudaDeviceSynchronize(); 
    iElaps = cpuSecond() - iStart; 
    cudaMemcpy(h_odata, d_odata, grid.x/8*sizeof(int), cudaMemcpyDeviceToHost); 
    gpu_sum = 0; 
    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i]; 
    printf("gpu Template reduceUnrolling8v2 elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", iElaps,gpu_sum,grid.x/8,block.x);




    // free host memory 
    free(h_idata); 
    free(h_odata); 
    // free device memory 
    cudaFree(d_idata); 
    cudaFree(d_odata); 
    // reset device 
    cudaDeviceReset(); 
    // check the results 
    bResult = (gpu_sum == cpu_sum); 
    if(!bResult) printf("Test failed!\n"); 
    return EXIT_SUCCESS; 
}