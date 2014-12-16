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

__global__ void GPU_Update(glm::mat4 *modelMatrices, glm::vec3 *d, glm::vec3 *dj, glm::vec3 *c, glm::vec3 *raxis, float *w, int n, float cT) {
        int i = GPU_globalindex();
        if(i < n)
        {

                float epsilon = 1.0e-4;

                dj[i].x = -sinf(cT);
                dj[i].y = 0.00;
                dj[i].z = cosf(cT);

                float theta = 0.0;
                glm::vec3 cr(0,0,0);
                if(glm::length(d[i]-dj[i]) > epsilon)
                {
                        theta = glm::acos(glm::dot(glm::normalize(d[i]),glm::normalize(dj[i])));
                        cr = glm::normalize(glm::cross(d[i],dj[i]));
                }

                if(glm::length(raxis[i]) > epsilon)
                {
                        modelMatrices[i] = glm::rotate(modelMatrices[i], -w[i], raxis[i]);
                }
                modelMatrices[i] = glm::translate(modelMatrices[i],dj[i]*0.0125f);               // Falta Delta, reemp. por 0.0125
                if(glm::length(raxis[i]) > epsilon)
                {
                        modelMatrices[i] = glm::rotate(modelMatrices[i], w[i], raxis[i]);
                }


                if(glm::length(cr) > epsilon)
                {
                        modelMatrices[i] = glm::rotate(modelMatrices[i], theta, cr);
                        raxis[i] = glm::normalize(glm::cross(glm::vec3(0.0,0.0,0.25),dj[i]));
                        w[i] = glm::acos(glm::dot(glm::normalize(dj[i]),glm::normalize(glm::vec3(0.0,0.0,0.25))));;
                }

                d[i] = dj[i];
                c[i] += d[i]*0.0125f;

        }
}

void update(glm::mat4 *modelMatrices, glm::vec3 *d, glm::vec3 *dj, glm::vec3 *c, glm::vec3 *raxis, float *w, int n, float cT) {
        glm::mat4 *d_modelMatrices;
        glm::vec3 *d_d, *d_dj, *d_c, *d_raxis;
        float *d_w;

        size_t m4size = n * sizeof(glm::mat4);
        size_t v3size = n * sizeof(glm::vec3);
        size_t fsize = n * sizeof(float);
        
        cudaMalloc(&d_modelMatrices, m4size);
        cudaMalloc(&d_d, v3size);
        cudaMalloc(&d_dj, v3size);
        cudaMalloc(&d_c, v3size);
        cudaMalloc(&d_raxis, v3size);
        cudaMalloc(&d_w, fsize);

        cudaMemcpy(d_modelMatrices, modelMatrices, m4size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_d, d, v3size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_dj, dj, v3size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_c, c, v3size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_raxis, raxis, v3size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_w, w, fsize, cudaMemcpyHostToDevice);

        dim3 grid(n,1,1);           // Max 2147483647 , 65535, 65535 blocks
        dim3 block(1,1,1);          // Max 1024 threads per block
        GPU_Update<<<grid,block>>> (d_modelMatrices, d_d, d_dj, d_c, d_raxis, d_w, n, cT);
        
        cudaMemcpy(modelMatrices, d_modelMatrices, m4size, cudaMemcpyDeviceToHost);
        cudaMemcpy(d, d_d, v3size, cudaMemcpyDeviceToHost);
        cudaMemcpy(dj, d_dj, v3size, cudaMemcpyDeviceToHost);
        cudaMemcpy(c, d_c, v3size, cudaMemcpyDeviceToHost);
        cudaMemcpy(raxis, d_raxis, v3size, cudaMemcpyDeviceToHost);
        cudaMemcpy(w, d_w, fsize, cudaMemcpyDeviceToHost);
        
        cudaFree(d_modelMatrices);
        cudaFree(d_d);
        cudaFree(d_dj);
        cudaFree(d_c);
        cudaFree(d_raxis);
        cudaFree(d_w);
}