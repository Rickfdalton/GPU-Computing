/*
This is the parallel implementation of matrix multiplication of NXN matrix


*/
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>


using namespace std;

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
__global__ void mat_mul_kernel(float* d_A, float* d_B, float* d_C,int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i<N && j<N ){
        float val = 0;
        for (int k=0; k< N; k++){
            val+=d_A[i*N+k]*d_B[k*N+j];
        }
        d_C[i*N+j]=val;
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
    dim3 numThreadsPerBlock(16,16,1);
    dim3 numBlocksPerGrid((N+16-1)/16, (N+16-1)/16, 1);

    timer.start();
    mat_mul_kernel<<<numBlocksPerGrid,numThreadsPerBlock>>>(d_A,d_B,d_C,N);
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
    int N = 100;
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

    for(int i=0;i<N;i++){
          for(int j=0;j<N;j++){
            cout << C_parallel[i*N +j] -  C[i*N +j] << flush;
          }
          cout<< endl;
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_parallel;


    return 0;
}