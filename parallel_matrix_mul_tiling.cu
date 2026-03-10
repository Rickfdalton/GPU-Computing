/*
This is the parallel implementation of matrix multiplication of NXN matrix
using tiling method.
elapsed time CPU 4111.5
elapsed time GPU 4.62031
Max_Diff 9.15527e-05

*/
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

// define as macro. this get resolved before compiling so no need to worry about copying to gpu 
#define TILE_DIM 32

/* Timer class to get the elapsed time*/
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
Timer timer; //initiate timer

//kernel impl
__global__ void mat_mul_tile_kernel(float* d_A, float* d_B, float* d_C,int N){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // this is not global this is shared per block
    __shared__ float A_s[TILE_DIM][TILE_DIM]; 
    __shared__ float B_s[TILE_DIM][TILE_DIM];

    float sum=0; // go in a register.

    //per iteration we load corresponding A tile and B tile
    //each thread (corr: each output element) loads no_tiles number of elements in A and no_tiles number of elements in B
    unsigned int no_tiles =  (N + TILE_DIM - 1) / TILE_DIM;
    for (unsigned int tile=0; tile< no_tiles; tile++){
            A_s[threadIdx.y][threadIdx.x] = (row <N && (tile * TILE_DIM+ threadIdx.x)< N )? d_A[row * N + (tile * TILE_DIM+ threadIdx.x) ] :0.0f;
            B_s[threadIdx.y][threadIdx.x] = (col <N && (tile * TILE_DIM+ threadIdx.y)< N )? d_B[(tile * TILE_DIM+ threadIdx.y)*N+ col] : 0.0f;

            __syncthreads();
       
            for (unsigned int i=0; i< TILE_DIM ; i++){
                sum+= A_s[threadIdx.y][i]*B_s[i][threadIdx.x];
            }
            __syncthreads();
    }
    if(row < N && col < N){
        d_C[row * N + col] = sum;
    }
    
}

void parallel_matrix_multiply(float* A, float*B, float*C, int N){
    //allocate GPU memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A,N*N*sizeof(float));
    cudaMalloc((void**)&d_B,N*N*sizeof(float));
    cudaMalloc((void**)&d_C,N*N*sizeof(float));

    //copy to GPU memory
    cudaMemcpy(d_A,A,N*N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,N*N*sizeof(float),cudaMemcpyHostToDevice);


    //kernel call
    dim3 numThreadsPerBlock(TILE_DIM,TILE_DIM,1);
    dim3 numBlocksPerGrid((N+TILE_DIM-1)/TILE_DIM, (N+TILE_DIM-1)/TILE_DIM, 1);

    timer.start();
    mat_mul_tile_kernel<<<numBlocksPerGrid,numThreadsPerBlock>>>(d_A,d_B,d_C,N);
    cudaDeviceSynchronize();
    timer.stop();
    timer.print("GPU");

    //copy data from GPU
    cudaMemcpy(C,d_C,N*N*sizeof(float),cudaMemcpyDeviceToHost);


    //free GPU mem
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


}

void matrix_multiply(float* A, float* B, float* C, int N){
   for(int row =0; row< N; row++){
     for(int col =0; col< N; col++){
        int idx = row*N + col;
        float val=0;
        for (int k=0;k<N;k++){
            val+=A[row*N + k]*B[k*N+col];
        }
        C[idx]=val;
     }
   }
}

int main(){
    // cudaDeviceSynchronize();
    int N = 1000;
    float* A = new float[N*N];
    float* B = new float[N*N];
    float* C = new float[N*N];
    float* C_parallel = new float[N*N];


    for(int i=0; i< N*N; i++){
        A[i]=drand48();
        B[i]=drand48();
        C[i]=0;
        C_parallel[i]=0;

    }

    timer.start();
    matrix_multiply(A,B,C,N);
    timer.stop();
    timer.print("CPU");

    parallel_matrix_multiply(A,B,C_parallel,N);

    float max_diff= 0.0f;
    for(int i=0;i<N;i++){
          for(int j=0;j<N;j++){
            max_diff = std::max (std::abs(C_parallel[i*N +j] -  C[i*N +j]) , max_diff);
          }
    }
    cout<< "Max_Diff "<< max_diff <<endl;

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_parallel;

    return 0;
}