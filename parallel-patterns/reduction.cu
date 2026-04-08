/*
Parallel pattern 3 : reduction
*/

#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_DIM 1024
using namespace std;

__global__ void reduction_kernel(float* input_d, float* partial_sums_d, unsigned int N){
    __shared__ float input_block_s[2* BLOCK_DIM];

    //store to shared memory
    if((2*blockDim.x * blockIdx.x + (threadIdx.x * 2))< N){
        input_block_s[threadIdx.x * 2]=input_d[2*blockDim.x * blockIdx.x + (threadIdx.x * 2)];
    }else{
        input_block_s[threadIdx.x * 2]=0.0f;
    } 
    if((2*blockDim.x * blockIdx.x + (threadIdx.x * 2)+1)< N){
        input_block_s[1+ threadIdx.x * 2]=input_d[2*blockDim.x * blockIdx.x + (threadIdx.x * 2)+1];
    }else{
        input_block_s[1+ threadIdx.x * 2]=0.0f;
    }

    __syncthreads();

 
    for(int stride=1; stride< 2*blockDim.x; stride*=2){
        int idx = 2*stride*threadIdx.x;
        if (idx < 2 * blockDim.x) {
        input_block_s[idx]+=input_block_s[idx +stride];
        }
        __syncthreads();
    }

    partial_sums_d[blockIdx.x]=input_block_s[0];

}

float reduce_gpu(float* input, unsigned int N){
    //allocate memory
    float* input_d;
    cudaMalloc((void**)&input_d, N*sizeof(float));
    cudaMemcpy(input_d, input,N*sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = BLOCK_DIM;
    int elementsPerBlock = 2*threadsPerBlock;
    int blocksPerGrid = (N+elementsPerBlock-1)/elementsPerBlock;

    //allocate partial sums
    float* partial_sums = (float*) malloc(blocksPerGrid*sizeof(float));
    float* partial_sums_d;
    cudaMalloc((void**)&partial_sums_d,blocksPerGrid*sizeof(float));

    //call kernel
    reduction_kernel<<<blocksPerGrid,threadsPerBlock>>>(input_d,partial_sums_d,N);

    //copy back
    cudaMemcpy(partial_sums,partial_sums_d,blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);

    //reduce partial sums in cpu
    float sum = 0.0f;
    for(int i=0;i<blocksPerGrid;i++){
        sum+=partial_sums[i];
    }

    //free
    free(partial_sums);
    cudaFree(input_d);
    cudaFree(partial_sums_d);

    return sum;

}

int main(){

    const int N= 2048;
    float input[N];

    for(int i=0; i< N; i++){
        input[i]=1.0;
    }

    float sum= reduce_gpu(input, N);

    std::cout<< "sum" << sum << endl;

    return 0;
}