#include <cuda_runtime.h>
#include <cmath>

__global__ void kernel_1(int N) 
{ 
    double sum = 0.0; 
    for (int i = 0; i < N; i++) 
    { 
        sum = sum + tan(0.1) * tan(0.1); 
    } 
}
__global__ void kernel_2(int N) 
{ 
    double sum = 0.0; 
    for (int i = 0; i < N; i++) 
    { 
        sum = sum + tan(0.1) * tan(0.1); 
    } 
}
__global__ void kernel_3(int N) 
{ 
    double sum = 0.0; 
    for (int i = 0; i < N; i++) 
    { 
        sum = sum + tan(0.1) * tan(0.1); 
    } 
}
__global__ void kernel_4(int N) 
{ 
    double sum = 0.0; 
    for (int i = 0; i < N; i++) 
    { 
        sum = sum + tan(0.1) * tan(0.1); 
    } 
}