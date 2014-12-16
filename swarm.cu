#define GLM_FORCE_RADIANS

#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/norm.hpp>

#include "main.h"

const float FLOCKSIZE = 5.0;
#define ATTRACTION_VELOCITY 1.0f
#define ORIENTATION_VELOCITY 0.5f
const float epsilon = 1.0e-4;

// __device__ Functions

__device__ int GPU_globalindex(){
        return  blockIdx.z * gridDim.y * gridDim.x * blockDim.z * blockDim.y * blockDim.x +
                blockIdx.y * gridDim.x * blockDim.z * blockDim.y * blockDim.x +
                blockIdx.x * blockDim.z * blockDim.y * blockDim.x +
                threadIdx.z * blockDim.y * blockDim.x +
                threadIdx.y * blockDim.x +
                threadIdx.x;
}

__device__ void closest_neighbors(int*& points_indices, int& n_points, int global_index, int total_number, glm::vec3 *positions){
  n_points = 0;
  for(int j=0;j<total_number;++j){
    if(j != global_index){
      if(glm::distance(positions[j], positions[global_index]) < FLOCKSIZE){
        points_indices[n_points] = j;
        n_points++;
      }
    }
  }
}

// __global__ Functions

__device__ glm::vec3 resultant(glm::vec3* positions, int* points_indices, int n_points, int cur_boid_index){
    glm::vec3 new_vec(0,0,0);
    for(int j=0;j<n_points;++j)
        new_vec += glm::normalize(positions[points_indices[j]] - positions[cur_boid_index]);
    return new_vec;
}

__device__ glm::vec3 resultant_direction(glm::vec3* directions, int* points_indices, int n_points, int cur_boid_index){
    glm::vec3 new_vec(0,0,0);
    for(int j=0;j<n_points;++j)
        new_vec += glm::normalize(directions[points_indices[j]]);

    return new_vec;
}

__global__ void GPU_update_vector(glm::vec3 *directions_output, glm::vec3 *directions_input, glm::vec3 *positions, int num_boids){
        int index = GPU_globalindex();
        if(index < num_boids)
        {
                directions_output[index] = glm::vec3(0,0,0);
                int n_points = 0;
                int* points_indices = new int[num_boids];
                closest_neighbors(points_indices, n_points, index, num_boids, positions);
                glm::vec3 sum_vector = resultant(positions, points_indices, n_points, index);
                glm::vec3 sum_direction = resultant_direction(directions_input, points_indices, n_points, index);
                directions_output[index] += sum_vector*-1.0f;
                directions_output[index] += sum_vector*ATTRACTION_VELOCITY + sum_direction*ORIENTATION_VELOCITY;
                delete[] points_indices;
        }
}

__global__ void GPU_Update(glm::mat4 *modelMatrices, glm::vec3 *directions, glm::vec3 *updated_directions, glm::vec3 *positions, glm::vec3 *raxis, float *angles, int num_boids, float cT) {
        int i = GPU_globalindex();
        if(i < num_boids)
        {

                float theta = 0.0;
                glm::vec3 cr(0,0,0);
                if(glm::length(directions[i]-updated_directions[i]) > epsilon)
                {
                        theta = glm::acos(glm::dot(glm::normalize(directions[i]),glm::normalize(updated_directions[i])));
                        cr = glm::normalize(glm::cross(directions[i],updated_directions[i]));
                }

                if(glm::length(raxis[i]) > epsilon)
                {
                        modelMatrices[i] = glm::rotate(modelMatrices[i], -angles[i], raxis[i]);
                }
                modelMatrices[i] = glm::translate(modelMatrices[i],updated_directions[i]*0.0125f);               // Falta Delta, reemp. por 0.0125
                if(glm::length(raxis[i]) > epsilon)
                {
                        modelMatrices[i] = glm::rotate(modelMatrices[i], angles[i], raxis[i]);
                }


                if(glm::length(cr) > epsilon)
                {
                        modelMatrices[i] = glm::rotate(modelMatrices[i], theta, cr);
                        raxis[i] = glm::normalize(glm::cross(glm::vec3(0.0,0.0,0.25),updated_directions[i]));
                        angles[i] = glm::acos(glm::dot(glm::normalize(updated_directions[i]),glm::normalize(glm::vec3(0.0,0.0,0.25))));;
                }

                directions[i] = updated_directions[i];
                positions[i] += directions[i]*0.0125f;

        }
}

void update(glm::mat4 *modelMatrices, glm::vec3 *directions,
            glm::vec3 *updated_directions, glm::vec3 *positions,
            glm::vec3 *raxis, float *angles, int num_boids, float cT) {
        glm::mat4 *d_modelMatrices;
        glm::vec3 *new_directions, *new_updated_directions, *new_positions, *new_raxis;
        float *new_angles;

        size_t m4size = num_boids * sizeof(glm::mat4);
        size_t v3size = num_boids * sizeof(glm::vec3);
        size_t fsize  = num_boids * sizeof(float);

        cudaMalloc(&d_modelMatrices, m4size);
        cudaMalloc(&new_directions, v3size);
        cudaMalloc(&new_updated_directions, v3size);
        cudaMalloc(&new_positions, v3size);
        cudaMalloc(&new_raxis, v3size);
        cudaMalloc(&new_angles, fsize);

        cudaMemcpy(d_modelMatrices, modelMatrices, m4size, cudaMemcpyHostToDevice);
        cudaMemcpy(new_directions, directions, v3size, cudaMemcpyHostToDevice);
        cudaMemcpy(new_updated_directions, updated_directions, v3size, cudaMemcpyHostToDevice);
        cudaMemcpy(new_positions, positions, v3size, cudaMemcpyHostToDevice);
        cudaMemcpy(new_raxis, raxis, v3size, cudaMemcpyHostToDevice);
        cudaMemcpy(new_angles, angles, fsize, cudaMemcpyHostToDevice);

        dim3 grid(num_boids,1,1);  // Max 2147483647 , 65535, 65535 blocks
        dim3 block(1,1,1);          // Max 1024 threads per block
        GPU_update_vector<<<grid,block>>> (
                        new_updated_directions,
                        new_directions,
                        new_positions, num_boids);
        GPU_Update<<<grid,block>>> (
                        d_modelMatrices, new_directions,
                        new_updated_directions, new_positions,
                        new_raxis, new_angles, num_boids, cT);

        cudaMemcpy(modelMatrices, d_modelMatrices, m4size, cudaMemcpyDeviceToHost);
        cudaMemcpy(directions, new_directions, v3size, cudaMemcpyDeviceToHost);
        cudaMemcpy(updated_directions, new_updated_directions, v3size, cudaMemcpyDeviceToHost);
        cudaMemcpy(positions, new_positions, v3size, cudaMemcpyDeviceToHost);
        cudaMemcpy(raxis, new_raxis, v3size, cudaMemcpyDeviceToHost);
        cudaMemcpy(angles, new_angles, fsize, cudaMemcpyDeviceToHost);

        cudaFree(d_modelMatrices);
        cudaFree(new_directions);
        cudaFree(new_updated_directions);
        cudaFree(new_positions);
        cudaFree(new_raxis);
        cudaFree(new_angles);
}
