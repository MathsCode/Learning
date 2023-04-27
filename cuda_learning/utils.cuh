#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <stdlib.h>
#include <time.h>


double cpuSecond() 
{ 
    struct timeval tp; 
    gettimeofday(&tp,NULL); 
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6); 
}

void fill_data(float *data,int tot,int low,int high)
{
    srand(time(nullptr));
    for(int i = 0; i < tot; i++)
    {
        data[i] = low + (rand() % ((high-low) * 100))/100.0;
    }
}
void fill_data(int *data,int tot,int low,int high)
{
    srand(time(nullptr));
    for(int i = 0; i < tot; i++)
    {
        data[i] = low + rand() % (high-low);
    }
}

void GetDeviceBasicInfo(int dev = 0)
{
    cudaDeviceProp deviceProp; 
    cudaGetDeviceProperties(&deviceProp, dev); 
    printf("%d starting reduction at ", dev); 
    printf("device %d: %s ", dev, deviceProp.name); 
    cudaSetDevice(dev); 
}