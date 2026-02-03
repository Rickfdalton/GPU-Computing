/*
This is the parallel implementation of Image blur

elapsed time CPU 42.328
elapsed time GPU 0.246729

*/

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include "stb_image.h"
#include "stb_image_write.h"


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

//kernel implementation
__global__ void blur_image_kernel(unsigned char* img, unsigned char* blur_d, unsigned int width,unsigned int height ){
         int i = blockIdx.x * blockDim.x +threadIdx.x;
         int j = blockIdx.y * blockDim.y +threadIdx.y;

        if (i>0 && i< width-1 && j>0 && j<height-1){

        //get the index of current and nearby pixel
        int idx_top_left = (j-1) * width + (i-1);
        int idx_top = (j-1) * width + i;
        int idx_top_right = (j-1) * width + (i+1);
        int idx_left = j * width + (i-1);
        int idx = j * width + i;
        int idx_right = j * width + (i+1);
        int idx_bottom_left = (j+1) * width + i-1;
        int idx_bottom = (j+1) * width + i;
        int idx_bottom_right = (j+1) * width + i+1;

        int red = 0;
        int green = 0;
        int blue = 0;

        red = (img[idx_top_left*3]+img[idx_top*3]+img[idx_top_right*3]+img[idx_left*3]+img[idx*3]+img[idx_right*3]+img[idx_bottom_left*3]+img[idx_bottom*3]+img[idx_bottom_right*3])/9;
        green = (img[idx_top_left*3 +1]+img[idx_top*3 +1]+img[idx_top_right*3+1]+img[idx_left*3+1]+img[idx*3+1]+img[idx_right*3+1]+img[idx_bottom_left*3+1]+img[idx_bottom*3+1]+img[idx_bottom_right*3+1])/9;
        blue = (img[idx_top_left*3+2]+img[idx_top*3+2]+img[idx_top_right*3+2]+img[idx_left*3+2]+img[idx*3+2]+img[idx_right*3+2]+img[idx_bottom_left*3+2]+img[idx_bottom*3+2]+img[idx_bottom_right*3+2])/9;

        blur_d[idx*3] = static_cast<unsigned char>(red);
        blur_d[idx*3 +1] = static_cast<unsigned char>(green);
        blur_d[idx*3 +2] = static_cast<unsigned char>(blue);

        }

}


void parallel_image_blur(unsigned char* img, unsigned char* img_blur, unsigned int width, unsigned int height){
    //alocalte GPU memory
    unsigned char* image_d;
    unsigned char* blur_d;
    cudaMalloc((void**)&image_d, width*height*3*sizeof(unsigned char));
    cudaMalloc((void**)&blur_d, width*height*3*sizeof(unsigned char));

    //copy to GPU memory
    cudaMemcpy(image_d,img,width*height*3*sizeof(char), cudaMemcpyHostToDevice);

    //call kernel
    unsigned int no_threads= (width) *(height);
    dim3 threads_per_block(32,32,1);
    dim3 blocks_per_grid(( (width)+threads_per_block.x -1)/threads_per_block.x,((height)+threads_per_block.y -1)/threads_per_block.y);

    timer.start();
    blur_image_kernel<<<blocks_per_grid, threads_per_block>>>( image_d,  blur_d, width, height);
    cudaDeviceSynchronize();
    timer.stop();
    timer.print("GPU");

    //copy from GPU memory
    cudaMemcpy(img_blur,blur_d,width*height*3*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    //free memory
    cudaFree(image_d);
    cudaFree(blur_d);

}

int main(){
    // cudaDeviceSynchronize();

    int width, height, nrChannels;
    unsigned char *img = stbi_load("apple.jpg", &width, &height, &nrChannels, 0);
    if (!img){
        cout << "error loading image" << endl;
    }

    unsigned char *img_blur_parallel = new unsigned char[width * height * 3];// channel is 3

    parallel_image_blur(img,img_blur_parallel,width,height);


    stbi_write_png("parallel_blur.png", width, height, 3, img_blur_parallel, width * 3);
    
    stbi_image_free(img);
    delete[] img_blur_parallel;
    return 0;
}