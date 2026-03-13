/*
This is the optimized parallel implementation of matrix multiplication of NXN matrix
using tiling method.
Here each thread is not assigned to each output element, but
each thread is assigned to N output elemetns in same row which basicallly sits in same position in each grid.

[t_0][][]  [t_0][][]    [t_0][][]
[][][]     [][][]        [][][]
[][][]     [][][]        [][][]

elapsed time CPU 4048.18
elapsed time GPU 2.99956
Max_Diff 9.15527e-05

*/
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

// define as macro. this get resolved before compiling so no need to worry about copying to gpu 
#define TILE_DIM 32
#define COARSE_FACTOR 4

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
__global__ void mat_mul_tile_kernel_opt(float* d_A, float* d_B, float* d_C,int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col[COARSE_FACTOR];

    for(unsigned int col_no=0;col_no < COARSE_FACTOR ; col_no++ ){
        col[col_no]= blockIdx.x * blockDim.x * COARSE_FACTOR  + threadIdx.x + TILE_DIM * col_no;
    }

    //shared per block
    __shared__ float A_s[TILE_DIM][TILE_DIM]; // load A
    __shared__ float B_s[TILE_DIM][TILE_DIM * COARSE_FACTOR]; //load B

    float sum[COARSE_FACTOR]= {0}; // go in a register.

    //we load corresponding A tile and (COARSEFACTOR * B) tiles
    unsigned int no_tiles =  (N + TILE_DIM - 1) / TILE_DIM;
    for (unsigned int tile=0; tile< no_tiles; tile++){
            A_s[threadIdx.y][threadIdx.x] = (row <N && (tile * TILE_DIM+ threadIdx.x)< N )? d_A[row * N + (tile * TILE_DIM+ threadIdx.x) ] :0.0f;
            for (unsigned int col_no=0; col_no < COARSE_FACTOR ; col_no++){
                B_s[threadIdx.y][threadIdx.x + col_no * TILE_DIM] = (col[col_no] <N && (tile * TILE_DIM+ threadIdx.y)< N )? d_B[(tile * TILE_DIM+ threadIdx.y)*N+ col[col_no]] : 0.0f;
            }
            __syncthreads();
       
            for (unsigned int i=0; i< TILE_DIM ; i++){
                for (int col_no=0; col_no < COARSE_FACTOR ; col_no++){
                    sum[col_no]+= A_s[threadIdx.y][i]*B_s[i][col_no * TILE_DIM+threadIdx.x];
                }
            }
            __syncthreads();
    }
    
    for (int col_no=0; col_no < COARSE_FACTOR ; col_no++){
        if(row < N && col[col_no] < N){
            d_C[row * N + col[col_no]] = sum[col_no];
        }
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
    dim3 numBlocksPerGrid((N+TILE_DIM-1)/(TILE_DIM*COARSE_FACTOR), (N+TILE_DIM-1)/TILE_DIM, 1);

    timer.start();
    mat_mul_tile_kernel_opt<<<numBlocksPerGrid,numThreadsPerBlock>>>(d_A,d_B,d_C,N);
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