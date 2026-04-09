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

/*
Driver code - claude generated
*/
int main(){
    // Test parameters
    int N = 10000000;  // 10 million elements
    
    // Allocate host memory
    float* input = (float*)malloc(N * sizeof(float));
    
    // Initialize with values (1.0f for easy verification)
    for(int i = 0; i < N; i++){
        input[i] = 1.0f;  // Sum should be N
    }
    
    // GPU reduction
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    float gpu_sum = reduce_gpu(input, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    
    // CPU reduction for verification
    float cpu_sum = 0.0f;
    for(int i = 0; i < N; i++){
        cpu_sum += input[i];
    }
    
    // Verify results
    printf("N = %d\n", N);
    printf("CPU Sum: %.2f\n", cpu_sum);
    printf("GPU Sum: %.2f\n", gpu_sum);
    printf("Error: %.6f\n", fabs(cpu_sum - gpu_sum));
    printf("GPU Time: %.3f ms\n", gpu_time);
    
    if(fabs(cpu_sum - gpu_sum) < 1e-3){
        printf("✓ PASSED!\n");
    } else {
        printf("✗ FAILED!\n");
    }
    
    // Cleanup
    free(input);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}