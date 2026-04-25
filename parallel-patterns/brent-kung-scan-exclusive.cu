#include <iostream>
#include <cuda_runtime.h>
#define BLOCK_DIM 1024

/*
Exclusive scan version of brent kung.
Only the post reduction step varies.
*/

__global__ void scan_kernel(float* input_d, float* output_d, float* partial_sums_d, int N){

    int i = 2*blockDim.x * blockIdx.x + threadIdx.x;

    //load to shared_mem
    __shared__ float input_s[2*BLOCK_DIM];
    if(i < N){
        input_s[threadIdx.x] = input_d[i];
    }else{
        input_s[threadIdx.x] = 0.0f;
    }

    if(i+BLOCK_DIM < N){
        input_s[threadIdx.x + BLOCK_DIM] = input_d[i+BLOCK_DIM];
    }else{
        input_s[threadIdx.x + BLOCK_DIM] = 0.0f;
    }
    __syncthreads();

    //reduction

    for(int stride=1; stride<=BLOCK_DIM; stride*=2){
        int idx = (threadIdx.x+1)* 2 *stride -1;
        if (idx < 2*BLOCK_DIM){
            input_s[idx]+= input_s[idx- stride];
        }
        __syncthreads();
    }

    //Get last element of block and put in partial scan array
    if(threadIdx.x == 0 ){
        partial_sums_d[blockIdx.x]= input_s[2*BLOCK_DIM -1];
        input_s[2*BLOCK_DIM - 1] = 0;
    }
    __syncthreads();

    //post reduction
    for(int stride=BLOCK_DIM; stride>0; stride/=2){
        int idx = (threadIdx.x+1)* 2 *stride -1;
        if (idx < 2*BLOCK_DIM){
            float temp= input_s[idx-stride];
            input_s[idx-stride]=input_s[idx];
            input_s[idx]+= temp;
        }
        __syncthreads();
    }

    //add to output
    if(i< N){
        output_d[i]= input_s[threadIdx.x];
    }
    if(i+ BLOCK_DIM < N){
        output_d[i+BLOCK_DIM]= input_s[threadIdx.x+BLOCK_DIM];
    }
}

__global__ void add_kernel(float* output_d, float* scan_partial_sums_d, int N){
    int i = 2*blockDim.x * blockIdx.x + threadIdx.x;
    float to_add =0.0f;
    if(blockIdx.x >0){
        to_add = scan_partial_sums_d[blockIdx.x];
    }

    if(i<N) output_d[i]+=to_add;
    if(i+BLOCK_DIM<N) output_d[i+BLOCK_DIM]+=to_add;

}

void brent_kung_gpu(float* input_d , float* output_d, int N){
    int no_threads_per_block = BLOCK_DIM;
    int no_elements_per_block = 2*no_threads_per_block;
    int no_blocks_per_grid = (N+no_elements_per_block-1)/no_elements_per_block;

    //partial sums
    float* partial_sums_d;
    cudaMalloc((void**)&partial_sums_d, no_blocks_per_grid*sizeof(float));
    float* scan_partial_sums_d;
    cudaMalloc((void**)&scan_partial_sums_d,no_blocks_per_grid*sizeof(float));
    float* sum_d;
    cudaMalloc((void**)&sum_d,sizeof(float) );

    scan_kernel<<<no_blocks_per_grid,no_threads_per_block>>>(input_d, output_d, partial_sums_d, N);
    cudaDeviceSynchronize();
    scan_kernel<<<1,no_threads_per_block>>>(partial_sums_d, scan_partial_sums_d, sum_d, no_blocks_per_grid);
    cudaDeviceSynchronize();
    add_kernel<<<no_blocks_per_grid,no_threads_per_block>>>(output_d, scan_partial_sums_d, N);

    cudaFree(partial_sums_d);
    cudaFree(scan_partial_sums_d);
    cudaFree(sum_d);

}

// Driver code - claude genereated
int main(){
    int N = 100000;
    size_t size = N * sizeof(float);
    
    // Host arrays
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);
    float *h_expected = (float*)malloc(size);
    
    // Initialize input
    for(int i = 0; i < N; i++){
        h_input[i] = 1.0f;
    }
    
    // CPU reference
    h_expected[0] = 0.0f;
    for(int i = 1; i < N; i++){
        h_expected[i] = h_expected[i-1] + h_input[i-1];
    }
    
    // Device arrays
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    // Run GPU scan
    brent_kung_gpu(d_input, d_output, N);
    
    // Copy result back
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    
    // Verify
    bool correct = true;
    int errors = 0;
    for(int i = 0; i < N && errors < 10; i++){
        if(fabs(h_output[i] - h_expected[i]) > 1e-5){
            printf("Mismatch at %d: GPU=%.0f, CPU=%.0f\n", i, h_output[i], h_expected[i]);
            correct = false;
            errors++;
        }
    }
    
    if(correct){
        printf("✓ All tests passed! N=%d\n", N);
        printf("First 10: ");
        for(int i = 0; i < 10; i++) printf("%.0f ", h_output[i]);
        printf("\nLast 10: ");
        for(int i = N-10; i < N; i++) printf("%.0f ", h_output[i]);
        printf("\n");
    }else{
        printf("✗ Test failed!\n");
    }
    
    // Cleanup
    free(h_input);
    free(h_output);
    free(h_expected);
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
