/*
Parallel pattern2 : stencil
*/

#include <iostream>
#include <cuda_runtime.h>
#define BLOCK_DIM 8
#define C_0 1
#define C_1 1


__global__ void stencil_kernel(float* in_d, float* out_d, unsigned int N){
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    int z = blockDim.z*blockIdx.z + threadIdx.z;

    if(x>0 && x< N-1 && y>0 && y< N-1 && z>0 && z< N-1 ){
        float sum= 
            C_0 * (in_d[(z*N*N + y*N +x)])+
            C_1*(in_d[((z+1)*N*N + y*N +x)] +in_d[((z-1)*N*N + y*N +x)]+
            in_d[(z*N*N + (y+1)*N +x)] +in_d[(z*N*N + (y-1)*N +x)]+
            in_d[(z*N*N + y*N +(x+1))] +in_d[(z*N*N + y*N +(x-1))]);
        out_d[(z*N*N + y*N +x)]=sum;      
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
    dim3 threadsPerBlock(BLOCK_DIM,BLOCK_DIM,BLOCK_DIM);
    dim3 blocksPerGrid((N+BLOCK_DIM-1)/BLOCK_DIM,(N+BLOCK_DIM-1)/BLOCK_DIM,(N+BLOCK_DIM-1)/BLOCK_DIM);

    stencil_kernel<<<blocksPerGrid,threadsPerBlock>>>(in_d,out_d,N);

    //copy back to CPU
    cudaMemcpy(out,out_d,N*N*N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(in_d);
    cudaFree(out_d);

}

int main(){
    int N =512;
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