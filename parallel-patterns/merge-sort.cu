#include <iostream>
#include <cuda_runtime.h>


__device__ void merge_seq(float* A, float* B, float* C,   int m,   int n);
__device__   int coRank(float* A, float* B,   int m,   int n,   int k);
__device__ void merge(float* A, float* B, float* C,   int m,   int n);
__global__ void merge_sort_kernel(float* src, float* dst, int N, int width);

/* merging 2 sorted array */
__device__ void merge_seq(float* A, float* B, float* C,   int m,   int n){
      int i=0;
      int j=0;
      int k=0;

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
__device__   int coRank(float* A, float* B,   int m,   int n,   int k){
      int low = max(k-n,0);
      int high = min(k,m);

    while (true){
          int i = (low+high)/2;
          int j = k-i;
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
#define BLOCK_SIZE 1024


__device__ void merge(float* A, float* B, float* C,   int m,   int n){

    extern __shared__ float shared[];
    float* A_s = shared;
    float* B_s = shared + m;
    float* C_s = shared + m + n;
  
    for(int i= threadIdx.x; i<m; i+=blockDim.x){
        A_s[i]=A[i];
    }

    for(int j= threadIdx.x; j<n; j+=blockDim.x){
        B_s[j]=B[j];
    }
    __syncthreads();

    // merge
    int k = threadIdx.x;

    if(k < m + n){
          int i = coRank(A_s, B_s, m,n,k);
          int j = k - i;
          int kNext = (k + 1 ) < m+n ? k+ 1 : m+n ;
          int iNext = coRank(A_s,B_s,m,n,kNext);
          int jNext = kNext - iNext;
        merge_seq(&A_s[i], &B_s[j], &C_s[k],iNext-i,jNext-j);
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
            n
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

    for(int level=1; level< N; level*=2){
    int blocks_per_grid = (N+2*level -1)/(2*level) ;
        merge_sort_kernel <<<blocks_per_grid, BLOCK_SIZE , 4*level*sizeof(float)>>> (A_d, C_d, N, level);
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
Driver code 
*/
int main(){
    int N = 16;

    float *A = new float[N];

    // Initialize sorted arrays
    for(  int i = 0; i < N; i++){
        A[i] = (N-i) * 2.0f;      
    }

    // Run GPU merge
    merge_sort_gpu(A,N);

    // Print result
    std::cout << "Sorted array:\n";
    for(  int i = 0; i < N; i++){
        std::cout << A[i] << " ";
    }
    std::cout << "\n";

    delete[] A;
    return 0;
}