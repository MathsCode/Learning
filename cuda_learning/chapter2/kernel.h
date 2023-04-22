#include <cuda_runtime.h>
#include <stdio.h>
__global__ void warmingup(float *c) 
{ 
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    float a, b; a = b = 0.0f; 
    int warpSize = 32;
    if ((tid / warpSize) % 2 == 0) 
    { 
        a = 100.0f; 
    } 
    else 
    { 
        b = 200.0f; 
    } 
    c[tid] = a + b; 
}

__global__ void mathKernel1(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a ,b;
    a = b = 0.0f;
    if(tid % 2 == 0)
    {
        a = 100.0f;
    }
    else
    {
        b = 200.0f;
    }
    c[tid] = a+b;
}

__global__ void mathKernel2(float *c) 
{ 
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    float a, b; a = b = 0.0f; 
    int warpSize = 32;
    if ((tid / warpSize) % 2 == 0) 
    { 
        a = 100.0f; 
    } 
    else 
    { 
        b = 200.0f; 
    } 
    c[tid] = a + b; 
}
__global__ void mathKernel3(float *c) 
{ 
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    float ia, ib; ia = ib = 0.0f; 
    bool ipred = (tid % 2 == 0); 
    if (ipred) 
    { 
        ia = 100.0f; 
    } 
    if (!ipred) 
    { 
        ib = 200.0f; 
    } 
    c[tid] = ia + ib; 
}
