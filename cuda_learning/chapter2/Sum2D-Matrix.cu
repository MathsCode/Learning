#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <iostream>
#define CHECK(call)                                     \
{                                                       \
    const cudaError_t error = call;                     \
    if(error != cudaSuccess)                            \
    {                                                   \
        printf("Error: %s:%d,",__FILE__,__LINE__);      \
        printf("code:%d,reason:%s\n",error,cudaGetErrorString(error)); \
        exit(1);                                        \
    }                                                   \
}

void initialize(float *A, float * B,int nElem)
{
    for(int i = 0; i < nElem; i++)
    {
        A[i] = (rand() % 10) * 0.1;
        B[i] = (rand() % 10) * 0.1;
    }
}
bool check_result(float *C_ref,float *C,int nElem)
{
    float epsilon = 1>>5;
    for(int i = 0; i < nElem; i++)
    {
        if(abs(C_ref[i] - C[i]) > epsilon)
        {
            // std::cout<<i<<' '<<C_ref[i]<<' '<<C[i]<<"\n";
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
void cal_host(float *A,float *B,float *C,int nx,int ny)
{
    for(int i = 0; i < ny; i++)
    {
        for(int j = 0; j < nx; j++)
        {
            C[j] = A[j] + B[j];
        }
        A += nx;
        B += nx;
        C += nx;
    }
}

__global__ void cal_device(float *A,float *B,float *C,int nx,int ny)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = iy * nx + ix;
    if(ix < nx && iy < ny)
    {
        C[idx] = A[idx] + B[idx];
        // printf("%f+%f=%f",A[idx],B[idx],C[idx]);
    }
}
int main()
{
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);
    // cudaDeviceReset();
    int nx = 1<<10;
    int ny = 1<<10;
    int nelem = nx*ny;
    size_t nbytes = nelem*sizeof(float);
    float *A_h=nullptr,*B_h=nullptr,*C_h=nullptr,*C_h_ref=nullptr;
    A_h = (float *)malloc(nbytes);
    B_h = (float *)malloc(nbytes);
    C_h = (float *)malloc(nbytes);
    C_h_ref = (float *)malloc(nbytes);
    initialize(A_h,B_h,nelem);
    double start,Elaps;
    start = cpuSecond();
    cal_host(A_h,B_h,C_h_ref,nx,ny);
    Elaps = cpuSecond() - start;

    std::cout<<Elaps<<std::endl;




    float *A_d = nullptr,*B_d = nullptr,*C_d = nullptr;
    cudaMalloc((void **)&A_d,nbytes);
    cudaMalloc((void **)&B_d,nbytes);
    cudaMalloc((void **)&C_d,nbytes);

    cudaMemcpy(A_d,A_h,nbytes,cudaMemcpyHostToDevice);
    cudaMemcpy(B_d,B_h,nbytes,cudaMemcpyHostToDevice);
    int dimx = 16;
    int dimy = 16;
    dim3 block(dimx,dimy);
    dim3 grid((nx+dimx-1)/dimx,(ny+dimy-1)/dimy);
    printf("<<<%d,%d>>>\n",(nx+dimx-1)/dimx,(ny+dimy-1)/dimy);
    // int warm_iter = 100;
    // for(int i = 0; i < warm_iter+10; i++)
    // {
    //     if(i == warm_iter)
    //     {
    //         start = cpuSecond();
    //     }
    //     cal_device<<<block,grid>>>(A_d,B_d,C_d,nx,ny);
    //     // cudaDeviceReset();
    //     cudaDeviceSynchronize();
        
    // }
    start = cpuSecond();
    (cal_device<<<block,grid>>>(A_d,B_d,C_d,nx,ny));
    cudaDeviceSynchronize();
    Elaps = cpuSecond() - start;
    std::cout<<Elaps<<std::endl;
    cudaMemcpy(C_h,C_d,nbytes,cudaMemcpyDeviceToHost);

    printf("%d\n",check_result(C_h_ref,C_h,nelem));
    

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    free(A_h);
    free(B_h);
    free(C_h);
    free(C_h_ref);
    cudaDeviceReset();

    
    
}