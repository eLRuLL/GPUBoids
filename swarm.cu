#define GLM_FORCE_RADIANS
#define FLOCKSIZE 10
#define MIN_COLLISON_AVOIDANCE 10
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/norm.hpp>
#include <stdio.h>

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

typedef struct
{
  float x, y, z;
} point;

__device__ float distance(glm::vec3 a, glm::vec3 b){
  return sqrtf( powf(a.x - b.x,2) + powf(a.y - b.y, 2) + powf(a.z - b.z, 2));
}

__device__ glm::vec3 substract(glm::vec3 a, glm::vec3 b){
  glm::vec3 rpta;
  rpta.x = a.x - b.x;
  rpta.y = a.y - b.y;
  rpta.z = a.z - b.z;
  return rpta;
}

__device__ glm::vec3 direction(glm::vec3 from, glm::vec3 to){
  return substract(to, from);
}

__global__ void GPU_hashmap(glm::vec3 *a, glm::vec3 *b, float *c, int n) {
        int index = GPU_globalindex();
        printf("en kernel");
        if(index < n)
        {
          //printf("Hello thread %d\n", index);

          //////////////////////////////////////////////////////////////////////////////////////////
          ////////// a es Direccion, b es posicion, esa era la idea, cambiale los nombres si gustas, c era 'angulo'
          //////////////////////////////////////////////////////////////////////////////////////////



                // a[index].x *= 1.05;
                // a[index].y *= 1.05;
                // a[index].z *= 1.05;
                int i;
                int cur_i = 0;
                glm::vec3* points = new glm::vec3[n];

                for(i=0;i<n;++i){
                  if(i != index)
                    if(distance(b[index], b[i]) < FLOCKSIZE){
                      points[cur_i] = b[i];
                      cur_i++;
                    }
                }
                printf("cur_i: %d, distance %f\n", cur_i, distance(b[index], b[i]));
                //printf("distance: %f\n", distance(b[index], b[i]));
                // printf("pos_actual: %f %f %f\n", b[index].x, b[index].y, b[index].z);

                for(i=0;i<cur_i;++i){
                  double dist = distance(b[index], points[i]);
                  if(dist < MIN_COLLISON_AVOIDANCE){
                    glm::vec3 temp = substract(b[index], direction(b[index], points[i]));
                    //printf("Hello thread %d %d %d\n", temp.x, temp.y, temp.z);
                    a[index].x = temp.x;
                    a[index].y = temp.y;
                    a[index].z = temp.z;
                  }
                }
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

void update(glm::vec3 *a, glm::vec3 *b, float *c, int n) {
        printf("calling UPDATE\n");
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
        GPU_hashmap<<<grid,block>>> (d_a, d_b, d_c, n);

        cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
}
