/*
convolution!... but with tiling.
*/

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>
#include "stb_image.h"
#include "stb_image_write.h"

#define OUT_TILE_DIM 16
#define MASK_RADIUS 2
#define MASK_DIM (MASK_RADIUS*2+1)
#define BLOCK_DIM (OUT_TILE_DIM+2*MASK_RADIUS)

using namespace std;

__constant__ float mask_c[MASK_DIM][MASK_DIM];


//kernel implementation
__global__ void tiled_convolution_kernel(unsigned char* image_d, unsigned char* img_out_d,unsigned int width, unsigned int height){
    __shared__ float image_s[BLOCK_DIM][3*BLOCK_DIM];

    int col = OUT_TILE_DIM* blockIdx.x + threadIdx.x;
    int row = OUT_TILE_DIM* blockIdx.y + threadIdx.y;

    col =col- MASK_RADIUS;
    row =row- MASK_RADIUS;


    for (unsigned int channel=0; channel<3; channel++){
        if (col >= 0 && col < width   && row >= 0 && row < height ){
        //load to shared memory
            image_s[threadIdx.y][3*threadIdx.x + channel] = image_d[3*(row*width + col)+channel];
        }else {
            image_s[threadIdx.y][3*threadIdx.x + channel] = 0.0f; 
        }
    }

    //sync
    __syncthreads();

    if (col >= 0 && col < width   && row >= 0 && row < height ){
            for (unsigned int channel=0; channel<3; channel++){
                /*  calculation is not for all the tile elements. so not all threads will now calculating its pixel
                    now we calculate only for tile (which is smaller than our block)
                    now control divergence comes in
                */
                if (threadIdx.x >= (MASK_RADIUS) && threadIdx.x < (BLOCK_DIM - MASK_RADIUS) && threadIdx.y>= (MASK_RADIUS ) && threadIdx.y < (BLOCK_DIM - MASK_RADIUS)){
                        float sum=0;
                        for(unsigned int idx=0; idx<MASK_DIM*MASK_DIM; idx++){
                            unsigned int i = idx / MASK_DIM ; //mask row
                            unsigned int j = idx % MASK_DIM ; //mask col

                            int i_neighbour = threadIdx.y + (i-MASK_RADIUS); //row
                            int j_neighbour = threadIdx.x + (j-MASK_RADIUS) ; //col

                            sum+= image_s[i_neighbour][3*j_neighbour + channel] * mask_c[i][j];
                            
                        }
                        sum = fminf(255.0f, fmaxf(0.0f, sum));
                        img_out_d[3*(row*width+col)+channel] = static_cast<unsigned char>(sum) ;
                }
            }
    }

    //sync
     __syncthreads();
    }

void convolution(float mask[][MASK_DIM], unsigned char* img, unsigned char* img_out, unsigned int width, unsigned int height){
    //alocalte GPU memory
    unsigned char* image_d;
    unsigned char* img_out_d;
    cudaMalloc((void**)&image_d, width*height*3*sizeof(unsigned char));
    cudaMalloc((void**)&img_out_d, width*height*3*sizeof(unsigned char));

    //copy to GPU memory
    cudaMemcpy(image_d,img,width*height*3*sizeof(char), cudaMemcpyHostToDevice);

    //copy mask to constant memory
    cudaMemcpyToSymbol(mask_c,mask, MASK_DIM * MASK_DIM* sizeof(float));

    //call kernel
    dim3 threads_per_block(BLOCK_DIM,BLOCK_DIM,1);
    dim3 blocks_per_grid(( (width)+OUT_TILE_DIM -1)/OUT_TILE_DIM,((height)+OUT_TILE_DIM -1)/OUT_TILE_DIM);

    tiled_convolution_kernel<<<blocks_per_grid, threads_per_block>>>( image_d,  img_out_d, width, height);
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

    float mask[MASK_DIM][MASK_DIM] = {
        1/25.0, 1/25.0, 1/25.0, 1/25.0, 1/25.0,
        1/25.0, 1/25.0, 1/25.0, 1/25.0, 1/25.0,
        1/25.0, 1/25.0, 1/25.0, 1/25.0, 1/25.0,
        1/25.0, 1/25.0, 1/25.0, 1/25.0, 1/25.0,
        1/25.0, 1/25.0, 1/25.0, 1/25.0, 1/25.0
    };

    convolution(mask,img,img_out,width,height);

    stbi_write_png("tiled-convoluted-image.png", width, height, 3, img_out, width * 3);
    
    stbi_image_free(img);
    delete[] img_out;
    return 0;
}