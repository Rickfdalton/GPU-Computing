#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <algorithm>

__device__   int coRank(float* A, float* B,   int m,   int n,   int k);
__device__ void merge(float* A, float* B, float* C,   int m,   int n);
__global__ void merge_sort_kernel(float* src, float* dst, int N, int width);


/*
Finding i from k
ie. when the index of the output array given find the correct index of input array A.
from that we can find the index of the input array B.
*/
__device__   int coRank(float* A, float* B,   int m,   int n,   int k){
      int low = max(k-n,0);
      int high = min(k,m);

    while (low < high){
          int i = (low+high)/2;
          int j = k-i;
        if(i>0 && j<n && A[i-1]>B[j]){
            high=i;
        }else if(j>0 && i<m && B[j-1]>A[i]){
            low=i+1;
        }else{
            return i;
        }
    }return low;
}

/*
merge kernel
each thread is responsible for calculating element for each index of output array
*/
#define BLOCK_SIZE 1024


__device__ void merge(float* A, float* B, float* C,   int m,   int n, int stride){

    extern __shared__ float shared[];
    float* A_s = shared;
    float* B_s = shared + m;
    float* C_s = shared + m + n;
  
    for(int i= threadIdx.x; i<m; i+=blockDim.x){
        A_s[i]=A[i];
    }
    __syncthreads();

    for(int j= threadIdx.x; j<n; j+=blockDim.x){
        B_s[j]=B[j];
    }
    __syncthreads();

    // merge
    int k = threadIdx.x;

    if(k < m + n){
          int i = coRank(A_s, B_s, m,n,k);
          int j = k - i;
          C_s[k] = (j == n || (i < m && A_s[i] <= B_s[j])) ? A_s[i] : B_s[j];
    }
    __syncthreads();

    //write to global memory
    for(int k= threadIdx.x; k<m+n; k+=blockDim.x){
        C[k]=C_s[k];
    }
}

__global__ void merge_sort_kernel(float* src, float* dst, int N, int width){
    int left = blockIdx.x * 2 * width;
    int right = left + width;

    if (right < N){
        int m = width;
        int n = min(width, N-right);
        merge(
            &src[left],
            &src[right],
            &dst[left],
            m,
            n,
            width
        );
    }else{
        for(int i = threadIdx.x; i < min(width, N-left); i += blockDim.x)
            dst[left + i] = src[left + i];
        return;
    }
}

void merge_sort_gpu(float* A,int N){
    float* A_d;
    float* C_d;

    cudaMalloc((void**) &A_d, N*sizeof(float));
    cudaMalloc((void**) &C_d, N*sizeof(float));

    cudaMemcpy(A_d, A, N*sizeof(float), cudaMemcpyHostToDevice);

    for(int stride=1; stride< N; stride*=2){
    int blocks_per_grid = (N+2*stride -1)/(2*stride) ;
        merge_sort_kernel <<<blocks_per_grid, BLOCK_SIZE , 4*stride*sizeof(float)>>> (A_d, C_d, N, stride);
        float *temp = C_d;
        C_d = A_d;
        A_d = temp;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(A, A_d, N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(C_d);
}


/*
Driver code - LLM generated
*/

int main(){
    int N = 1001;
    float *A = new float[N];

    // Initialize random array
    for(int i = 0; i < N; i++){
        A[i] = (float)drand48() * 2000.0f - 1000.0f;
    }

    std::cout << "Before sort:\n";
    for(int i = 0; i < 20; i++){
        std::cout << A[i] << " ";
    }
    std::cout << "\n\n";

    // Run GPU merge sort
    merge_sort_gpu(A, N);

    std::cout << "After sort:\n";
    for(int i = 0; i < 20; i++){
        std::cout << A[i] << " ";
    }
    std::cout << "\n";

    // Check if sorted using std::is_sorted
    bool is_sorted = std::is_sorted(A, A + N);
    std::cout << "Array is " << (is_sorted ? "sorted" : "not sorted") << "\n";

    delete[] A;
    return 0;
}
