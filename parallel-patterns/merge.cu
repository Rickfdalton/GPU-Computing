/*
Parallelizing merging of 2 sorted arrays
*/
#include <iostream>
#include <cuda_runtime.h>


__device__ void merge_seq(float* A, float* B, float* C, unsigned int m, unsigned int n);
__device__ unsigned int coRank(float* A, float* B, unsigned int m, unsigned int n, unsigned int k);
__global__ void merge_kernel(float* A, float* B, float* C, unsigned int m, unsigned int n);

/* merging 2 sorted array */
__device__ void merge_seq(float* A, float* B, float* C, unsigned int m, unsigned int n){
    unsigned int i=0;
    unsigned int j=0;
    unsigned int k=0;

    while(i<m and j<n){
        if(A[i]<=B[j]){
            C[k++]=A[i++];
        }else{
            C[k++]=B[j++];
        }
    }
    while(i<m){
        C[k++]=A[i++];
    }
    while(j<n){
        C[k++]=B[j++];
    }
}

/*
Finding i from k
ie. when the index of the output array given find the correct index of input array A.
from that we can find the index of the input array B.
*/
__device__ unsigned int coRank(float* A, float* B, unsigned int m, unsigned int n, unsigned int k){
    unsigned int low = max(k-n,0);
    unsigned int high = min(k,m);

    while (true){
        unsigned int i = (low+high)/2;
        unsigned int j = k-i;
        if(i>0 && j<n && A[i-1]>B[j]){
            high=i;
        }else if(j>0 && i<m && B[j-1]>A[i]){
            low=i;
        }else{
            return i;
        }
    }
}

/*
merge kernel
each thread is responsible for calculating element for each index of output array
*/
#define ELEM_PER_THREAD 6
#define THREADS_PER_BLOCK 128
#define ELEM_PER_BLOCK (ELEM_PER_THREAD*THREADS_PER_BLOCK)

__global__ void merge_kernel(float* A, float* B, float* C, unsigned int m, unsigned int n){
    unsigned int c = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int k = c* ELEM_PER_THREAD;

    if(k < m+n){
        unsigned int i = coRank(A, B, m,n,k);
        unsigned int j = k - i;
        unsigned int kNext = (k + ELEM_PER_THREAD ) < m+n ? k+ ELEM_PER_THREAD : m+n ;
        unsigned int iNext = coRank(A,B,m,n,kNext);
        unsigned int jNext = kNext - iNext;
        merge_seq(&A[i], &B[j], &C[k],iNext-i,jNext-j);
    }
    
}

void merge_gpu(float* A, float* B, float* C, unsigned int m, unsigned int n){
    float* A_d;
    float* B_d;
    float* C_d;

    cudaMalloc((void**) &A_d, m*sizeof(float));
    cudaMalloc((void**) &B_d, n*sizeof(float));
    cudaMalloc((void**) &C_d, (m+n)*sizeof(float));

    cudaMemcpy(A_d, A, m*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, n*sizeof(float), cudaMemcpyHostToDevice);

    unsigned int blocks_per_grid = (m+n+ELEM_PER_BLOCK -1)/ELEM_PER_BLOCK ;
    merge_kernel <<<blocks_per_grid, THREADS_PER_BLOCK>>> (A_d, B_d, C_d, m,n);

    cudaMemcpy(C, C_d, (m+n)*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

/*
Driver code : chatGPT generated
*/
int main(){
    unsigned int m = 16;
    unsigned int n = 16;

    float *A = new float[m];
    float *B = new float[n];
    float *C = new float[m + n];

    // Initialize sorted arrays
    for(unsigned int i = 0; i < m; i++){
        A[i] = i * 2.0f;        // 0,2,4,...
    }

    for(unsigned int i = 0; i < n; i++){
        B[i] = i * 2.0f + 1.0f; // 1,3,5,...
    }

    // Run GPU merge
    merge_gpu(A, B, C, m, n);

    // Print result
    std::cout << "Merged array:\n";
    for(unsigned int i = 0; i < m + n; i++){
        std::cout << C[i] << " ";
    }
    std::cout << "\n";

    // Verify sorted
    bool sorted = true;
    for(unsigned int i = 1; i < m + n; i++){
        if(C[i] < C[i-1]){
            sorted = false;
            break;
        }
    }

    if(sorted){
        std::cout << "✅ Merge successful\n";
    } else {
        std::cout << "❌ Merge failed\n";
    }

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}