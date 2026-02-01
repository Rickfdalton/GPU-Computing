#include <iostream>
#include "timer.h"
using namespace std;

void add_vectors_CPU(float* A, float* B, float* C, unsigned int N ){
    for(unsigned int i=0; i<N; i++){
        C[i]=A[i]+B[i];
    }
}


int main(){
    unsigned int N = 5;
    Timer timer;

    float* A = (float*) malloc(N*sizeof(float));
    float* B = (float*) malloc(N*sizeof(float));
    float* C = (float*) malloc(N*sizeof(float));

    for(unsigned int i=0; i< N ; i++){
        A[i]= rand();
        B[i]= rand();
    }

    timer.start();
    add_vectors_CPU(A,B,C,N);
    timer.stop();
    timer.print();

    free(A);
    free(B);
    free(C);

    return 0;
}