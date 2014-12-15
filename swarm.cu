#define GLM_FORCE_RADIANS
#include <iostream>
#include <stdio.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/norm.hpp>

#include "main.h"

// __device__ Functions

__device__ int GPU_globalindex(){
        return  blockIdx.z * gridDim.y * gridDim.x * blockDim.z * blockDim.y * blockDim.x + 
                blockIdx.y * gridDim.x * blockDim.z * blockDim.y * blockDim.x + 
                blockIdx.x * blockDim.z * blockDim.y * blockDim.x + 
                threadIdx.z * blockDim.y * blockDim.x +
                threadIdx.y * blockDim.x + 
                threadIdx.x;
}

// _global_ Functions

// __global__ void GPU_hashmap(glm::vec3 *a, int n) {
//         int i = GPU_globalindex();
//         if(i < n)
//         {
//                 // a[i].x = int(a[i].x)+9;
//                 // a[i].y = int(a[i].y)+9;
//                 // a[i].z = int(a[i].z)+9;

//                 a[i].x *= 1.05;
//                 a[i].y *= 1.05;
//                 a[i].z *= 1.05;
//         }
// }

__global__ void GPU_hashmap(glm::vec3 *a, glm::vec3 *b, float *c, int n, float cT) {
        printf("En kernel");
        int i = GPU_globalindex();
        if(i < n)
        {
                a[i].x = /*((i%2)*2-1)**/0.5*cosf(cT*0.5);
                a[i].y = -0.5;
                a[i].z = /*((i%2)*2-1)**/0.5*sinf(cT*0.5);

                // a[i].x = 0.05*(cosf(cT)-cT*sinf(cT));
                // a[i].y = -0.50;
                // a[i].z = 0.05*(cT*cosf(cT)+sinf(cT));

                // a[i].x = 0.25*(cT>1.0)-0.50*(cT>2.0);
                // a[i].y = 0.25*(cT>1.0)-0.50*(cT>2.0);
                // a[i].z = 0.25;

                // a[i].x = 0;
                // a[i].y = 0.25;
                // a[i].z = 0.25;

                // c[i] *= 1.05;
        }
}

// __global__ void GPU_inverse(float *a, int n) {
//         int i = GPU_globalindex();
//         if(i < n)
//                 a[i] = 255-a[i];
// }

// __global__ void GPU_grayscale(pix_t *a, pix_t *b, int n, int c) {
//         int i = GPU_globalindex();

//         if(i < n)
//         {
//                 i *= c;
//                 int grey_value = 0;
                
//                 for(int j=0; j<c; j++)
//                         grey_value += a[i+j];
//                 grey_value /= c;

//                 for(int j=0; j<c; j++)
//                         b[i+j] = grey_value;
//         }
// }

// __global__ void GPU_binarize(pix_t *a, int n, pix_t thresh) {
//         int i = GPU_globalindex();
//         if(i < n)
//                 a[i] = (a[i] >= thresh)*255;
// }

// main functions

// void update(glm::vec3 *a, int n) {
//         glm::vec3 *d_a;
//         size_t size = n * sizeof(glm::vec3);
//         dim3 grid(n,1,1);           // Max 2147483647 , 65535, 65535 blocks
//         dim3 block(1,1,1);          // Max 1024 threads per block
        
//         cudaMalloc(&d_a, size);
//         cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
//         GPU_hashmap<<<grid,block>>> (d_a, n);
//         cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);
//         cudaFree(d_a);
// }

void update(glm::vec3 *a, glm::vec3 *b, float *c, int n, float cT) {
        std::cout << "Aca estoy \n";
        glm::vec3 *d_a;
        glm::vec3 *d_b;
        float *d_c;

        size_t size = n * sizeof(glm::vec3);
        
        cudaMalloc(&d_a, size);
        cudaMalloc(&d_b, size);
        cudaMalloc(&d_c, n * sizeof(float));

        cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_c, c, n * sizeof(float), cudaMemcpyHostToDevice);

        dim3 grid(n,1,1);           // Max 2147483647 , 65535, 65535 blocks
        dim3 block(1,1,1);          // Max 1024 threads per block
        GPU_hashmap<<<grid,block>>> (d_a, d_b, d_c, n, cT);
        
        cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
}