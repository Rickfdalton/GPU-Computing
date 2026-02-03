/*
This is the GPU implementation of RGB to Grayscale conversion

note: when running in colab T4 GPU set the architectue to -arch=sm_75 to avoid kernel launch failures
results:
elapsed time GPU 0.160975
elapsed time CPU 5.66352

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
__global__ void rgb_to_gray_kernel(unsigned char* red, unsigned char* green, unsigned char* blue,unsigned char* gray, int width, int height ){
    unsigned int x = ((blockIdx.x) * (blockDim.x))+ threadIdx.x ;
    unsigned int y = ((blockIdx.y) * (blockDim.y)) + threadIdx.y ;
    if (x < width && y < height) {
        unsigned int idx = y*width + x;
        gray[idx] = static_cast<unsigned char>(
        0.299f*red[idx] + 0.587f*green[idx] + 0.114f*blue[idx]);
    }

}

void  parallel_RGB_gray(unsigned char *img ,unsigned char* gray, int width,int height) {

    //get individual channels
    unsigned char *red   = new unsigned char[width*height];
    unsigned char *green = new unsigned char[width*height];
    unsigned char *blue  = new unsigned char[width*height];

    for (int i=0; i< width * height ; i++){
        red[i]= img[3*i + 0];
        green[i]=img[3*i + 1];
        blue[i]=img[3*i + 2];
    }

    //allocate GPU memory
    unsigned char *red_d,*green_d,*blue_d, *gray_d;
    cudaMalloc((void**)&red_d,width*height*sizeof(unsigned char));
    cudaMalloc((void**)&green_d,width*height*sizeof(unsigned char));
    cudaMalloc((void**)&blue_d,width*height*sizeof(unsigned char));
    cudaMalloc((void**)&gray_d,width*height*sizeof(unsigned char));


    //copy GPU memory
    cudaMemcpy(red_d, red, width*height*sizeof(unsigned char) , cudaMemcpyHostToDevice );
    cudaMemcpy(green_d, green, width*height*sizeof(unsigned char) , cudaMemcpyHostToDevice );
    cudaMemcpy(blue_d, blue, width*height*sizeof(unsigned char) , cudaMemcpyHostToDevice );

    //call kernel
    timer.start();

    dim3 numThreadsPerBlock(16,16,1); // 32 x 32
    dim3 numBlocksPerGrid((width+numThreadsPerBlock.x - 1)/numThreadsPerBlock.x,(height+numThreadsPerBlock.y - 1)/numThreadsPerBlock.y,1); 
        

    rgb_to_gray_kernel<<<numBlocksPerGrid, numThreadsPerBlock>>>(red_d, green_d, blue_d, gray_d, width, height);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Launch failed: " << cudaGetErrorString(err) << std::endl;
        std::cout << "  Grid:  " << numBlocksPerGrid.x << " × " << numBlocksPerGrid.y << std::endl;
        std::cout << "  Block: " << numThreadsPerBlock.x << " × " << numThreadsPerBlock.y << std::endl;
        // exit(1);   // or return from function
    } else {
        std::cout << "Launch reported OK" << std::endl;
    }

    cudaDeviceSynchronize(); 
    timer.stop();
    timer.print("GPU");


    //copy data from GPU
    cudaMemcpy(gray, gray_d, width*height*sizeof(unsigned char) , cudaMemcpyDeviceToHost );


    //free mem
    cudaFree(red_d);
    cudaFree(green_d);
    cudaFree(blue_d);    
    delete[] red;
    delete[] green;
    delete[] blue;
}

int main(){
    // cudaDeviceSynchronize();

    int width, height, nrChannels;
    unsigned char *img = stbi_load("apple.jpg", &width, &height, &nrChannels, 0);
    if (!img){
        cout << "error loading image" << endl;
    }

    unsigned char *gray = new unsigned char[width * height];// channel is 1

    parallel_RGB_gray(img,gray,width,height);

    stbi_write_png("parallel_gray.png", width, height, 1 , gray, width);
    
    stbi_image_free(img);

    delete[] gray;
    return 0;
}