/*
tiled stencil.
*/

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

#define BLOCK_DIM 8
#define C_0 1
#define C_1 1

using namespace std;

__global__ void stencil_tiled_kernel(float* in_d, float* out_d, unsigned int N){
    int x = BLOCK_DIM *blockIdx.x + threadIdx.x -1;
    int y = BLOCK_DIM *blockIdx.y + threadIdx.y -1;
    int z = BLOCK_DIM *blockIdx.z + threadIdx.z -1;

    //load to shared memory
    __shared__ float in_s[BLOCK_DIM+2][BLOCK_DIM+2][BLOCK_DIM+2];

    if(x>=0 && x<N && y>=0 && y<N  && z>=0 && z<N  ){
        in_s[ threadIdx.z] [threadIdx.y][threadIdx.x]= 
        in_d[N*N*(z) +  N*(y) + (x)];
    }else {
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    if(threadIdx.x >0 && threadIdx.x <BLOCK_DIM+1 &&
       threadIdx.y >0 && threadIdx.y <BLOCK_DIM+1 &&
       threadIdx.z >0 && threadIdx.z <BLOCK_DIM+1 &&
       x>0 && y>0 && z>0 &&
       x<N-1 && y<N-1 && z<N-1 
    )
    {
        int local_x= threadIdx.x;
        int local_y= threadIdx.y;
        int local_z= threadIdx.z;

        float sum= 
            C_0 * (in_s[local_z][local_y][local_x])+
            C_1 * (in_s[local_z+1][local_y][local_x] + in_s[local_z-1][local_y][local_x]+
            in_s[local_z][local_y+1][local_x] + in_s[local_z][local_y-1][local_x]+
            in_s[local_z][local_y][local_x+1] + in_s[local_z][local_y][local_x-1]);
        
        out_d[(z)*N*N + (y)*N +x]=sum;      
    }    
}

void stencil_gpu(float* in, float* out, unsigned int N){
    float* in_d;
    float* out_d;

    //allocate memory and copy to GPU
    cudaMalloc((void**)&in_d, N*N*N*sizeof(float));
    cudaMalloc((void**)&out_d, N*N*N*sizeof(float));

    cudaMemcpy(in_d,in,N*N*N*sizeof(float), cudaMemcpyHostToDevice);
    
    //call kernel
    dim3 threadsPerBlock(BLOCK_DIM+2,BLOCK_DIM+2,BLOCK_DIM+2);
    dim3 blocksPerGrid((N+BLOCK_DIM-1)/BLOCK_DIM,(N+BLOCK_DIM-1)/BLOCK_DIM,(N+BLOCK_DIM-1)/BLOCK_DIM);

    stencil_tiled_kernel<<<blocksPerGrid,threadsPerBlock>>>(in_d,out_d,N);

    //copy back to CPU
    cudaMemcpy(out,out_d,N*N*N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(in_d);
    cudaFree(out_d);

}

int main(){
    int N =16;
    float* in = new float[N*N*N];
    float* out = new float[N*N*N];

    for (int i = 0; i < N * N * N; i++) {
        in[i] = 1.0f;  
        out[i] = 0.0f;
    }
    stencil_gpu(in,out,N);

    delete[] in;
    delete[] out;

    return 0;
}