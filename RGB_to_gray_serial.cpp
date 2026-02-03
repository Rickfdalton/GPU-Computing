/*
This is the serial implementation of RGB to Grayscale conversion
results:
elapsed time GPU 0.160975
elapsed time CPU 5.66352
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


void serial_RGB_gray(unsigned char* img, unsigned char* img_gray_serial, unsigned int width, unsigned int height){
    for (unsigned int i=0; i< width*height ; i++){
        unsigned char red = img[3*i + 0];
        unsigned char green = img[3*i + 1];
        unsigned char blue = img[3*i + 2];

        img_gray_serial[i] = static_cast<unsigned char>(
            0.299f*red + 0.587f*green + 0.114f*blue
        );
    }
}

int main(){
    // cudaDeviceSynchronize();

    int width, height, nrChannels;
    unsigned char *img = stbi_load("apple.jpg", &width, &height, &nrChannels, 0);
    if (!img){
        cout << "error loading image" << endl;
    }

    unsigned char *img_gray_serial = new unsigned char[width * height];// channel is 1

    timer.start();
    serial_RGB_gray(img,img_gray_serial,width,height);
    timer.stop();
    timer.print("CPU");

    stbi_write_png("serial_gray.png", width, height, 1 , img_gray_serial, width);
    

    stbi_image_free(img);
    delete[] img_gray_serial;
    return 0;
}