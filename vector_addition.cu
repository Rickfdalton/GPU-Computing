/*
Run on Google Colab GPU runtime T4 GPU
elapsed time CPU 1389.87 
elapsed time GPU 7.94866
*/


#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include "stb_image.h"

using namespace std;

class Timer{
public:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;

    void start(){
       start_time= std::chrono::high_resolution_clock::now();
    }
    void stop(){
        end_time= std::chrono::high_resolution_clock::now();
    }
    void print(const char* label){
        std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
        double ms = elapsed.count();
        std::cout<< "elapsed time "<< label<< " " << ms << std::endl;
    }
};
Timer timer;

__host__ __device__ float f(float a, float b){
    return a+b;
}

void add_vectors_CPU(float* A, float* B, float* C, unsigned int N ){
    for(unsigned int i=0; i<N; i++){
        C[i]=f(A[i],B[i]);
    }
}

//kernel (each thread will execute this)
    //gridDim.x : no of blocks per grid
    //blockDim.x : no of threads per block
    //blockIdx.x : position of block in grid
    //threadIdx.x : position of thread in block
// SPMD: single program multiple data
__global__ void vec_add_kernel(float* A, float* B, float* C, unsigned int N ){
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<N){
       C[i]=A[i]+B[i];
    }
}

void add_vectors_GPU(float* A, float* B, float* C, unsigned int N){
    //allocate GPU memory
    float* A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, N* sizeof(float));
    cudaMalloc((void**)&B_d, N* sizeof(float));
    cudaMalloc((void**)&C_d, N* sizeof(float));

    //copy to GPU
    cudaMemcpy(A_d, A, N* sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, N* sizeof(float),cudaMemcpyHostToDevice);

    //Call a GPU kernel functions (launch a grid of threads)
    // threads are organized in to blocks, collection of blocks is grid
    const unsigned int threads_per_block = 512;
    const unsigned int num_blocks = (N+ threads_per_block -1)/threads_per_block;
    timer.start();
    vec_add_kernel<<< num_blocks,threads_per_block >>>(A_d,B_d,C_d,N);
    cudaDeviceSynchronize();// wait for the kernel to finish
    timer.stop();
    timer.print("GPU");


    //copy from GPU
    cudaMemcpy(C, C_d,N* sizeof(float),cudaMemcpyDeviceToHost);

    //deallocate GPU mem
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);


}

int main(){
    cudaDeviceSynchronize();
    unsigned int N = 204800000;

    float* A = (float*) malloc(N*sizeof(float));
    float* B = (float*) malloc(N*sizeof(float));
    float* C = (float*) malloc(N*sizeof(float));

    for(unsigned int i=0; i< N ; i++){
        A[i]= rand();
        B[i]= rand();
    }

    timer.start();
    add_vectors_CPU(A,B,C,N);
    timer.stop();
    timer.print("CPU");


    add_vectors_GPU(A,B,C,N);
  

    free(A);
    free(B);
    free(C);

    return 0;
}