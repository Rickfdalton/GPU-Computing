/*
Radix Sort

Note that, iam using an inclusive scan so we need to treat the element fetching indices carefully
*/

#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <algorithm>

#define BLOCK_DIM 1024

using namespace std;

__global__ void segmented_scan_kernel(int* input_d, int* output_d ,int* partials_sums_d,int N, int blocks);
__global__ void stitch_scan(int* input, int* scan_of_partial_sums_d, int N, const int blocks);
__global__ void radix_sort_kernel( int* input,  int* output, int N);
void kogge_stone_inclusive_scan(int* input, int* output ,int N);
void radix_sort_gpu( int* input,  int* output, int N);

__global__ void segmented_scan_kernel(int* input_d, int* output_d ,int* partials_sums_d,int N, int blocks){

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* store to shared memory*/
    __shared__ int block_s[BLOCK_DIM];
    __shared__ int temp[BLOCK_DIM];


    if(i<N) output_d[i] = input_d[i];
    __syncthreads();

    if(i < N){
        block_s[threadIdx.x] = input_d[i];
    }else{
        block_s[threadIdx.x] = 0;
    }
    __syncthreads();

    for(int stride= 1; stride < BLOCK_DIM ; stride*=2){
        if (threadIdx.x >= stride)
            temp[threadIdx.x] = block_s[threadIdx.x] + block_s[threadIdx.x - stride];
        else
            temp[threadIdx.x] = block_s[threadIdx.x];

        __syncthreads();

        block_s[threadIdx.x] = temp[threadIdx.x];
        __syncthreads();
    }

    if (threadIdx.x == blockDim.x - 1 && i < N)
    partials_sums_d[blockIdx.x] = block_s[threadIdx.x];
    __syncthreads();

    if (i<N) output_d[i] = block_s[threadIdx.x];
}


__global__ void stitch_scan(int* input, int* scan_of_partial_sums_d, int N, const int blocks){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ int to_add;
    
    if(threadIdx.x ==0 && blockIdx.x ==0){
        to_add= 0;
    }else if(threadIdx.x ==0 && blockIdx.x >0){
        to_add= scan_of_partial_sums_d[blockIdx.x -1];
    }
    __syncthreads();

    if(i < N){
        input[i]+= to_add ;
    }

}

void kogge_stone_inclusive_scan(int* input, int* output ,int N){
    int threads_per_block = BLOCK_DIM;
    int blocks_per_grid = (N+threads_per_block-1)/threads_per_block;

    int* partials_sums_d;
    int* input_d;
    int* output_d;
    int* scan_of_partial_sums_d;
    int* total_sum;

    cudaMalloc((void**)&partials_sums_d,blocks_per_grid*sizeof(int));
    cudaMalloc((void**)&input_d,N*sizeof(int));
    cudaMalloc((void**)&output_d,N*sizeof(int));
    cudaMalloc((void**)&scan_of_partial_sums_d,blocks_per_grid* sizeof(int));
    cudaMalloc((void**)&total_sum,sizeof(int));


    cudaMemcpy(input_d, input, N*sizeof(int), cudaMemcpyHostToDevice);

    segmented_scan_kernel<<<blocks_per_grid,threads_per_block>>>(input_d,output_d,partials_sums_d, N, blocks_per_grid);
    cudaDeviceSynchronize();
    
    segmented_scan_kernel<<< 1, blocks_per_grid>>>(partials_sums_d,scan_of_partial_sums_d,total_sum, blocks_per_grid, 1);
    cudaDeviceSynchronize();

    stitch_scan <<<blocks_per_grid,threads_per_block>>>(output_d,scan_of_partial_sums_d, N ,blocks_per_grid);
    cudaDeviceSynchronize();

    cudaMemcpy(output, output_d, N*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(partials_sums_d);
    cudaFree(input_d);
    cudaFree(output_d);
    cudaFree(scan_of_partial_sums_d);
}

__global__ void bit_extraction_kernel( int* input, int* output, int N, int bit_idx){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i< N){
        int bit = (input[i]>> bit_idx)&1;
        output[i]= bit;
    }
}

__global__ void prepare_output_kernel( int* input, int* scan,  int* output, int N, int bit_idx){
    int i = blockDim.x* blockIdx.x + threadIdx.x;
    if(i<N){
        int bit = (input[i]>> bit_idx)&1;
        int no_ones_left = (i==0) ? 0 : scan[i-1];
        int out_zero = i - no_ones_left;
        int out_one = N - scan[N-1]+no_ones_left;
        
        if(bit ==1){
            output[out_one]=input[i];
        }else{
            output[out_zero]=input[i];
        }
    }
}

void radix_sort_gpu( int* input,  int* output, int N){
    //perform cpu max
     int max = input[0];
    for(int i =1; i< N;i++){
        if(input[i] > max){
            max=input[i];
        }
    }
    //find bit width
    int bit_width = 0;
    while(max>0){
        bit_width++;
        max>>=1;
    }

     int* input_d;
     int* output_d;
    int* bit_store_d;
    int* bit_scan_d;
    cudaMalloc((void**)&input_d, N*sizeof( int));
    cudaMalloc((void**)&output_d, N*sizeof( int));
    cudaMalloc((void**)&bit_store_d, N*sizeof(int));
    cudaMalloc((void**)&bit_scan_d, N*sizeof(int));

    cudaMemcpy(input_d, input, N*sizeof( int),cudaMemcpyHostToDevice);
    cudaMemset(bit_store_d,0,N*sizeof(int));
    cudaMemset(bit_scan_d,0,N*sizeof(int));

    int threads_per_block = BLOCK_DIM;
    int blocks_per_grid = (N+threads_per_block-1)/threads_per_block;

    for(int iter=0; iter<bit_width; iter++){
        bit_extraction_kernel<<<blocks_per_grid,threads_per_block>>>(input_d, bit_store_d,N,iter);
        kogge_stone_inclusive_scan(bit_store_d,bit_scan_d,N);
        prepare_output_kernel<<<blocks_per_grid,threads_per_block>>>( input_d, bit_scan_d, output_d,  N, iter);
        int* tmp = input_d;
        input_d = output_d;
        output_d = tmp;
    }

    cudaMemcpy(output, input_d, N*sizeof( int),cudaMemcpyDeviceToHost);

    cudaFree(input_d);
    cudaFree(output_d);
    cudaFree(bit_store_d);
    cudaFree(bit_scan_d);

}

/*
Driver code CHATGPT generated
*/
int main() {
    int N = 32;  // 🔥 keep small for debugging first

    vector<int> h_input(N);
    vector<int> h_output(N);
    vector<int> h_expected(N);

    // ----------------------------
    // 1. Generate test data
    // ----------------------------
    srand(0);
    cout << "Input:\n";
    for (int i = 0; i < N; i++) {
        h_input[i] = rand() % 100;
        cout << h_input[i] << " ";
        h_expected[i] = h_input[i];
    }
    cout << "\n\n";

    // ----------------------------
    // 2. CPU reference sort
    // ----------------------------
    sort(h_expected.begin(), h_expected.end());

    // ----------------------------
    // 3. Run GPU radix sort
    // ----------------------------
    radix_sort_gpu(h_input.data(), h_output.data(), N);

    // ----------------------------
    // 4. Print GPU result
    // ----------------------------
    cout << "GPU Output:\n";
    for (int i = 0; i < N; i++) {
        cout << h_output[i] << " ";
    }
    cout << "\n\n";

    // ----------------------------
    // 5. Print CPU expected
    // ----------------------------
    cout << "CPU Sorted:\n";
    for (int i = 0; i < N; i++) {
        cout << h_expected[i] << " ";
    }
    cout << "\n\n";

    // ----------------------------
    // 6. Validate
    // ----------------------------
    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (h_output[i] != h_expected[i]) {
            ok = false;
            cout << "Mismatch at " << i
                 << " GPU=" << h_output[i]
                 << " CPU=" << h_expected[i] << endl;
        }
    }

    if (ok)
        cout << "✅ Correct!\n";
    else
        cout << "❌ Wrong result!\n";

    return 0;
}