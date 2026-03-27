/*
Parallel Pattern 1 : convolution
*/

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include "stb_image.h"
#include "stb_image_write.h"

#define OUT_TILE_DIM 32
#define MASK_RADIUS 2
#define MASK_DIM (MASK_RADIUS*2+1)

using namespace std;

/*
constant memory
we can only allocate 64KB of constant memory
why constant memory is so small? we use SRAM for fast 
*/
__constant__ char mask_c[MASK_DIM][MASK_DIM];

//kernel implementation
__global__ void convolution_kernel(char* image_d, char* img_out_d,unsigned int width, unsigned int height){
    unsigned int col = blockDim.x* blockIdx.x + threadIdx.x;
    unsigned int row = blockDim.y* blockIdx.y + threadIdx.y;

    if (col < width && row < height){
        //loop through neighbouring elements and calculate weighted sum
        float sum=0;
        for(unsigned int idx=0; idx<MASK_DIM*MASK_DIM; idx++){
            unsigned int i = idx / MASK_DIM ; //row
            unsigned int j = idx % MASK_DIM ; //col

            int i_neighbour = row + (i-MASK_RADIUS); //row
            int j_neighbour = col + (j-MASK_RADIUS) ; //col

            if(i_neighbour>=0 && i_neighbour< height && j_neighbour>=0 && j_neighbour<width ){
                sum+= image_d[i_neighbour *width + j_neighbour] * mask_c[i][j];
            }
        }
        img_out_d[row*width+col] = static_cast<char>sum ;

    }
}

void convolution(char mask[][MASK_DIM], unsigned char* img, unsigned char* img_out, unsigned int width, unsigned int height){
    //alocalte GPU memory
    unsigned char* image_d;
    unsigned char* img_out_d;
    cudaMalloc((void**)&image_d, width*height*3*sizeof(unsigned char));
    cudaMalloc((void**)&img_out_d, width*height*3*sizeof(unsigned char));

    //copy to GPU memory
    cudaMemcpy(image_d,img,width*height*3*sizeof(char), cudaMemcpyHostToDevice);

    //copy mask to constant memory
    cudaMemcpyToSymbol(mask_c,mask, MASK_DIM * MASK_DIM* sizeof(char));

    //call kernel
    unsigned int no_threads= (width) *(height);
    dim3 threads_per_block(OUT_TILE_DIM,OUT_TILE_DIM,1);
    dim3 blocks_per_grid(( (width)+OUT_TILE_DIM -1)/OUT_TILE_DIM,((height)+OUT_TILE_DIM -1)/OUT_TILE_DIM);

    convolution_kernel<<<blocks_per_grid, threads_per_block>>>( image_d,  img_out_d, width, height);
    cudaDeviceSynchronize();

    //copy from GPU memory
    cudaMemcpy(img_out,img_out_d,width*height*3*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    //free memory
    cudaFree(image_d);
    cudaFree(img_out_d);

}

int main(){
    // cudaDeviceSynchronize();

    int width, height, nrChannels;
    unsigned char *img = stbi_load("apple.jpg", &width, &height, &nrChannels, 0);
    if (!img){
        cout << "error loading image" << endl;
    }

    unsigned char *img_out = new unsigned char[width * height * 3];// channel is 3

    convolution(img,img_out,width,height);

    stbi_write_png("convoluted-image.png", width, height, 3, img_out, width * 3);
    
    stbi_image_free(img);
    delete[] img_out;
    return 0;
}