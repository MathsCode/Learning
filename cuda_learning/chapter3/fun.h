#include <cuda_runtime.h>

int recursiveReduce(int *data,int const size)
{
    if(size == 1) return data[0];
    int stride = size / 2;
    for(int i = 0; i < stride; i++)
    {
        data[i] += data[i+stride];
    }
    return recursiveReduce(data,stride);
}


__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n) 
{ 
    // set thread ID 
    unsigned int tid = threadIdx.x; 
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // convert global data pointer to the local pointer of this block 
    int *idata = g_idata + blockIdx.x * blockDim.x; 
    // boundary check 
    if (idx >= n) return; 
    // in-place reduction in global memory 
    for (int stride = 1; stride < blockDim.x; stride *= 2) 
    { 
        if ((tid % (2 * stride)) == 0) 
        { 
            idata[tid] += idata[tid + stride]; 
        } 
        // synchronize within block 
        __syncthreads(); 
    } 
    // write result for this block to global mem 
    if (tid == 0) g_odata[blockIdx.x] = idata[0]; 
}

__global__ void reduceNeighboredLess (int *g_idata, int *g_odata, unsigned int n) 
{ 
    // set thread ID 
    unsigned int tid = threadIdx.x; 
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    // convert global data pointer to the local pointer of this block 
    int *idata = g_idata + blockIdx.x*blockDim.x; 
    // boundary check 
    if(idx >= n) return; 
    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) 
    { 
        // convert tid into local array index 
        int index = 2 * stride * tid; 
        if (index < blockDim.x) 
        { 
            idata[index] += idata[index + stride]; 
        } 
        // synchronize within threadblock 
        __syncthreads(); 
    } 
    // write result for this block to global mem 
    if (tid == 0) g_odata[blockIdx.x] = idata[0]; 
}

// Interleaved Pair Implementation with less divergence 
__global__ void reduceInterleaved (int *g_idata, int *g_odata, unsigned int n) 
{ 
    // set thread ID 
    unsigned int tid = threadIdx.x; 
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    // convert global data pointer to the local pointer of this block 
    int *idata = g_idata + blockIdx.x * blockDim.x;
    // boundary check 
    if(idx >= n) return; 
    // in-place reduction in global memory 
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) 
    { 
        if (tid < stride) 
        { 
            idata[tid] += idata[tid + stride]; 
        } 
        __syncthreads(); 
    } 
    // write result for this block to global mem 
    if (tid == 0) g_odata[blockIdx.x] = idata[0]; 
}
__global__ void reduceUnrolling2 (int *g_idata, int *g_odata, unsigned int n) 
{ 
    // set thread ID unsigned 
    int tid = threadIdx.x; 
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x; 
    // convert global data pointer to the local pointer of this block 
    int *idata = g_idata + blockIdx.x * blockDim.x * 2; 
    // unrolling 2 data blocks 
    if (idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];
    __syncthreads(); 
    // in-place reduction in global memory 
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) 
    { 
        if (tid < stride) 
        { 
            idata[tid] += idata[tid + stride]; 
        } 
        // synchronize within threadblock 
        __syncthreads(); 
    } 
    // write result for this block to global mem 
    if (tid == 0) g_odata[blockIdx.x] = idata[0]; 
}

__global__ void reduceUnrollWarps8 (int *g_idata, int *g_odata, unsigned int n) 
{ 
    // set thread ID 
    unsigned int tid = threadIdx.x; 
    unsigned int idx = blockIdx.x*blockDim.x*8 + threadIdx.x; 
    // convert global data pointer to the local pointer of this block 
    int *idata = g_idata + blockIdx.x*blockDim.x*8; 
    // unrolling 8 
    if (idx + 7*blockDim.x < n) 
    { 
        int a1 = g_idata[idx]; 
        int a2 = g_idata[idx+blockDim.x]; 
        int a3 = g_idata[idx+2*blockDim.x]; 
        int a4 = g_idata[idx+3*blockDim.x]; 
        int b1 = g_idata[idx+4*blockDim.x];
        int b2 = g_idata[idx+5*blockDim.x]; 
        int b3 = g_idata[idx+6*blockDim.x]; 
        int b4 = g_idata[idx+7*blockDim.x]; 
        g_idata[idx] = a1+a2+a3+a4+b1+b2+b3+b4;
    }
    __syncthreads();
    // in-place reduction in global memory 
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) 
    { 
        if (tid < stride) 
        { 
            idata[tid] += idata[tid + stride]; 
        } 
        // synchronize within threadblock 
        __syncthreads(); 
    } 
    // unrolling warp 
    if (tid < 32) 
    { 
        volatile int *vmem = idata; 
        vmem[tid] += vmem[tid + 32]; 
        vmem[tid] += vmem[tid + 16]; 
        vmem[tid] += vmem[tid + 8]; 
        vmem[tid] += vmem[tid + 4]; 
        vmem[tid] += vmem[tid + 2]; 
        vmem[tid] += vmem[tid + 1]; 
    } 
    // write result for this block to global mem 
    if (tid == 0) g_odata[blockIdx.x] = idata[0]; 
}

// Reducing with Complete Unrolling
__global__ void reduceUnrollWarps8v2 (int *g_idata, int *g_odata, unsigned int n) 
{ 
    // set thread ID 
    unsigned int tid = threadIdx.x; 
    unsigned int idx = blockIdx.x*blockDim.x*8 + threadIdx.x; 
    // convert global data pointer to the local pointer of this block 
    int *idata = g_idata + blockIdx.x*blockDim.x*8; 
    // unrolling 8 
    if (idx + 7*blockDim.x < n) 
    { 
        int a1 = g_idata[idx]; 
        int a2 = g_idata[idx+blockDim.x]; 
        int a3 = g_idata[idx+2*blockDim.x]; 
        int a4 = g_idata[idx+3*blockDim.x]; 
        int b1 = g_idata[idx+4*blockDim.x];
        int b2 = g_idata[idx+5*blockDim.x]; 
        int b3 = g_idata[idx+6*blockDim.x]; 
        int b4 = g_idata[idx+7*blockDim.x]; 
        g_idata[idx] = a1+a2+a3+a4+b1+b2+b3+b4;
    }
    __syncthreads();
    // in-place reduction in global memory 
    // unroll the blockdim
    // in-place reduction and complete unroll 
    if (blockDim.x>=1024 && tid < 512) idata[tid] += idata[tid + 512]; 
    __syncthreads(); 
    if (blockDim.x>=512 && tid < 256) idata[tid] += idata[tid + 256]; 
    __syncthreads(); 
    if (blockDim.x>=256 && tid < 128) idata[tid] += idata[tid + 128]; 
    __syncthreads(); 
    if (blockDim.x>=128 && tid < 64) idata[tid] += idata[tid + 64]; 
    __syncthreads();
    // unrolling warp 
    if (tid < 32) 
    { 
        volatile int *vmem = idata; 
        vmem[tid] += vmem[tid + 32]; 
        vmem[tid] += vmem[tid + 16]; 
        vmem[tid] += vmem[tid + 8]; 
        vmem[tid] += vmem[tid + 4]; 
        vmem[tid] += vmem[tid + 2]; 
        vmem[tid] += vmem[tid + 1]; 
    } 
    // write result for this block to global mem 
    if (tid == 0) g_odata[blockIdx.x] = idata[0]; 
}

//Reducing with Template Functions
//This will speed up at the compile state
template<unsigned int blocksize>
__global__ void reduceCompleteUnroll (int *g_idata, int *g_odata, unsigned int n) 
{ 
    // set thread ID 
    unsigned int tid = threadIdx.x; 
    unsigned int idx = blockIdx.x*blockDim.x*8 + threadIdx.x; 
    // convert global data pointer to the local pointer of this block 
    int *idata = g_idata + blockIdx.x*blockDim.x*8; 
    // unrolling 8 
    if (idx + 7*blockDim.x < n) 
    { 
        int a1 = g_idata[idx]; 
        int a2 = g_idata[idx+blockDim.x]; 
        int a3 = g_idata[idx+2*blockDim.x]; 
        int a4 = g_idata[idx+3*blockDim.x]; 
        int b1 = g_idata[idx+4*blockDim.x];
        int b2 = g_idata[idx+5*blockDim.x]; 
        int b3 = g_idata[idx+6*blockDim.x]; 
        int b4 = g_idata[idx+7*blockDim.x]; 
        g_idata[idx] = a1+a2+a3+a4+b1+b2+b3+b4;
    }
    __syncthreads();
    // in-place reduction in global memory 
    // unroll the blockdim
    // in-place reduction and complete unroll 
    if (blocksize>=1024 && tid < 512) idata[tid] += idata[tid + 512]; 
    __syncthreads(); 
    if (blocksize>=512 && tid < 256) idata[tid] += idata[tid + 256]; 
    __syncthreads(); 
    if (blocksize>=256 && tid < 128) idata[tid] += idata[tid + 128]; 
    __syncthreads(); 
    if (blocksize>=128 && tid < 64) idata[tid] += idata[tid + 64]; 
    __syncthreads();
    // unrolling warp 
    if (tid < 32) 
    { 
        volatile int *vmem = idata; 
        vmem[tid] += vmem[tid + 32]; 
        vmem[tid] += vmem[tid + 16]; 
        vmem[tid] += vmem[tid + 8]; 
        vmem[tid] += vmem[tid + 4]; 
        vmem[tid] += vmem[tid + 2]; 
        vmem[tid] += vmem[tid + 1]; 
    } 
    // write result for this block to global mem 
    if (tid == 0) g_odata[blockIdx.x] = idata[0]; 
}