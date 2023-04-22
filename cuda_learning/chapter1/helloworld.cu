#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>
#include <iostream>


void initialize(float *A, float * B,int nElem)
{
    for(int i = 0; i < nElem; i++)
    {
        A[i] = i*0.1;
        B[i] = i*0.1;
    }
}

void cal_host(float *A,float *B,float *C,int nElem)
{
    for(int i = 0; i < nElem; i++)
    {
        C[i] = A[i] + B[i];
    }
}

__global__ void cal_device(float *A,float *B,float *C,int nElem)
{
    int stride = blockDim.x;
    for(int i = threadIdx.x; i < nElem; i+=stride)
    {
        C[i] = A[i] + B[i];
    }
}

bool check_result(float *C_ref,float *C,int nElem)
{
    float epsilon = 1>>5;
    for(int i = 0; i < nElem; i++)
    {
        if(abs(C_ref[i] - C[i]) > epsilon)
        {
            return false;
        }
    }
    return true;
}

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}
int main()
{

    int dev = 0;
    cudaDeviceProp deviceprop;
    cudaGetDeviceProperties(&deviceprop,dev);
    std::cout<<dev<<' '<<deviceprop.name<<std::endl;
    cudaSetDevice(dev);
    cudaDeviceReset();
    int nElem;
    std::cin>>nElem;
    float *A_h = nullptr,*B_h = nullptr,*C_h_Ref = nullptr,*C_h = nullptr;
    int nBytes = nElem * sizeof(float);
    A_h = (float *)malloc(nBytes);
    B_h = (float *)malloc(nBytes);
    C_h_Ref = (float *)malloc(nBytes);
    C_h = (float *)malloc(nBytes);
    initialize(A_h,B_h,nElem);
    double start,Elaps;
    start = cpuSecond();
    cal_host(A_h,B_h,C_h_Ref,nElem );  
    Elaps = cpuSecond() - start;
    std::cout<<Elaps<<std::endl;

    float *A_d = nullptr,*B_d = nullptr,*C_d = nullptr;
    cudaMalloc((float **)&A_d,nBytes);
    cudaMalloc((float **)&B_d,nBytes);
    cudaMalloc((float **)&C_d,nBytes);

    cudaMemcpy(A_d,A_h,nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(B_d,B_h,nBytes,cudaMemcpyHostToDevice);
    int thread_per_block = 512;
    dim3 grid (1,1,1);
    dim3 block (min(thread_per_block,nElem),1,1);
    int warm_iter = 100;
    for(int i = 0; i < warm_iter+10; i++)
    {
        if(i == warm_iter)
        {
            start = cpuSecond();
        }
        cal_device<<<grid,block>>>(A_d,B_d,C_d,nElem);
        // cudaDeviceReset();
        cudaDeviceSynchronize();
        Elaps = cpuSecond() - start;
    }
    std::cout<<Elaps/10<<std::endl;
    cudaMemcpy(C_h,C_d,nBytes,cudaMemcpyDeviceToHost);

    printf("%d\n",check_result(C_h_Ref,C_h,nElem));
    // for(int i = 0; i < nElem; i++)
    // {
    //     std::cout<<C_h[i]<<' ';
    // }
    
}