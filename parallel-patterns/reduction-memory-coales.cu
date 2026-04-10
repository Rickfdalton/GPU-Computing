/*
Reduction but 
we dont use shared memory
we utilize memory coalescing
*/

/*
thoery: instead of adding elements by stride,
if 8 elements in block, the 4 threads will add first 4 elements with last 4 elements.
kernel time 633.02 micro seconds
*/

#include <iostream>
#include <cuda_runtime.h>
#define BLOCK_SIZE 1024

__global__ void reduction_kernel(float* input_d, float* partial_sum_d, int N){
    //fill block
    int idx=blockIdx.x* (2*blockDim.x)+ threadIdx.x;
    if(idx>= N){
        input_d[idx] = 0.0f;
    }
    if(idx + BLOCK_SIZE >= N){
        input_d[idx + BLOCK_SIZE] = 0.0f;
    }
    __syncthreads();

    for(int stride =BLOCK_SIZE; stride>0; stride/=2 ){
        if(threadIdx.x < stride){
            input_d[idx] += input_d[idx + stride];
        }
        __syncthreads();
    }
    
    if(threadIdx.x==0){
        partial_sum_d[blockIdx.x]= input_d[idx];
    }
}

float gpu_reduce(float* input, int N){
    int threads_per_block = BLOCK_SIZE;
    int elements_per_block = 2*threads_per_block;
    int blocks_per_grid = (N+elements_per_block-1)/elements_per_block;

    float* partial_sum = (float*)malloc(blocks_per_grid*sizeof(float));

    float* input_d;
    float* partial_sum_d;
    cudaMalloc((void**)&input_d, blocks_per_grid*elements_per_block*sizeof(float));
    cudaMalloc((void**)&partial_sum_d, blocks_per_grid*sizeof(float));
    cudaMemcpy(input_d,input,N*sizeof(float),cudaMemcpyHostToDevice);

    reduction_kernel<<<blocks_per_grid,threads_per_block>>>(input_d,partial_sum_d,N);
    cudaMemcpy(partial_sum,partial_sum_d,blocks_per_grid*sizeof(float),cudaMemcpyDeviceToHost);

    //cpu_reduce
    float sum=0.0f;
    for(int i=0;i<blocks_per_grid;i++){
        sum+=partial_sum[i];
    }

    cudaFree(partial_sum_d);
    cudaFree(input_d);
    free(partial_sum);

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
    float gpu_sum = gpu_reduce(input, N);
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