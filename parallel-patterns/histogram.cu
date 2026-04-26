/*
Histogram of pixel values of a image
*/
#define STB_IMAGE_IMPLEMENTATION

#include <iostream>
#include <cuda_runtime.h>
#include "../stb/stb_image.h"

#define NUM_BINS 256
#define BLOCK_DIM 16

using namespace std;


void histogram_gpu(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height);

__global__ void histogram_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height){

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = row * width + col;

    unsigned int pixel = 0;
    if(row<height && col<width){
        pixel = image[idx];
        atomicAdd(&bins[pixel], 1);
    }
}

void histogram_gpu(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height){

    unsigned char* image_d;
    unsigned int* bins_d;
    cudaMalloc((void**)&image_d, width*height*sizeof(unsigned char));
    cudaMalloc((void**)&bins_d, NUM_BINS*sizeof(unsigned int));

    cudaMemcpy(image_d,image,width*height*sizeof(unsigned char),cudaMemcpyHostToDevice);
    cudaMemset(bins_d,0,NUM_BINS*sizeof(unsigned int));

    dim3 blocks_per_grid = dim3((width+BLOCK_DIM-1)/BLOCK_DIM, (height+BLOCK_DIM-1)/BLOCK_DIM ,1);
    dim3 threads_per_block = dim3(BLOCK_DIM,BLOCK_DIM,1);

    histogram_kernel<<<blocks_per_grid,threads_per_block>>>(image_d, bins_d, width, height);

    cudaMemcpy(bins,bins_d,NUM_BINS*sizeof(unsigned int),cudaMemcpyDeviceToHost);

    cudaFree(image_d);
    cudaFree(bins_d);
}

int main(){

    int width, height, nrChannels;
    unsigned char *img = stbi_load("../apple.jpg", &width, &height, &nrChannels, 1);
    if (!img){
        cout << "error loading image" << endl;
        return -1;
    }

    unsigned int bins[NUM_BINS]={};

    histogram_gpu(img, bins, width, height);
    cudaDeviceSynchronize();

    for(int i=0; i<NUM_BINS; i++){
        std::cout<< i<<" "<< bins[i] << std::endl;
    }
    
    stbi_image_free(img);

    return 0;
}


