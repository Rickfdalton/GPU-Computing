/*
This is the serial implementation of Image blur
*/

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include <chrono>
// #include <cuda_runtime.h>
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


void serial_image_blur(unsigned char* img, unsigned char* img_blur_serial, unsigned int width, unsigned int height){
    for ( int i=1; i< width -1 ; i++){
        for( int j=1; j<height -1; j++)
        {                  
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

                red = (img[idx_top_left*3]+img[idx_top*3]+img[idx_top_right*3]+img[idx_left*3]+img[idx*3]+img[idx_right*3]+img[idx_bottom_left*3]+img[idx_bottom*3]+img[idx_bottom_right*3])* intensity /9;
                green = (img[idx_top_left*3 +1]+img[idx_top*3 +1]+img[idx_top_right*3+1]+img[idx_left*3+1]+img[idx*3+1]+img[idx_right*3+1]+img[idx_bottom_left*3+1]+img[idx_bottom*3+1]+img[idx_bottom_right*3+1])* intensity/9;
                blue = (img[idx_top_left*3+2]+img[idx_top*3+2]+img[idx_top_right*3+2]+img[idx_left*3+2]+img[idx*3+2]+img[idx_right*3+2]+img[idx_bottom_left*3+2]+img[idx_bottom*3+2]+img[idx_bottom_right*3+2])* intensity/9;

                img_blur_serial[idx*3] = static_cast<unsigned char>(red);
                img_blur_serial[idx*3 +1] = static_cast<unsigned char>(green);
                img_blur_serial[idx*3 +2] = static_cast<unsigned char>(blue);
       
        }
    }
}

int main(){
    // cudaDeviceSynchronize();

    int width, height, nrChannels;
    unsigned char *img = stbi_load("apple.jpg", &width, &height, &nrChannels, 0);
    if (!img){
        cout << "error loading image" << endl;
    }

    unsigned char *img_blur_serial = new unsigned char[width * height * 3];// channel is 3

    timer.start();
    serial_image_blur(img,img_blur_serial,width,height);
    timer.stop();
    timer.print("CPU");

    stbi_write_png("serial_blur.png", width, height, 3, img_blur_serial, width * 3);
    
    stbi_image_free(img);
    delete[] img_blur_serial;
    return 0;
}