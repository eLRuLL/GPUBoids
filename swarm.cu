#define GLM_FORCE_RADIANS

#include <iostream>
#include <stdio.h>
#include <cuda.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtx/vector_angle.hpp>

const float kRepulsionZoneRadius = 1.0;
const float kOrientationZoneRadius = 10.0;
const float kVisualFieldAngle = 180.0 / 180.0 * 3.14159;

#define ATTRACTION_WEIGHT 1.0f
#define ORIENTATION_WEIGHT 1.0f
#define PERCEIVED_CENTER_WEIGHT 10.0f
#define ACCELERATION 2.0f

// @DELTA should be the time from a frame to another
#define DELTA 0.1f
#define SIMULATION_SPEED 1.0f

const float epsilon = 1.0e-4;

// __device__ Functions

__device__ int GPU_globalindex() {
        return  blockIdx.z * gridDim.y * gridDim.x * blockDim.z * blockDim.y * blockDim.x +
                blockIdx.y * gridDim.x * blockDim.z * blockDim.y * blockDim.x +
                blockIdx.x * blockDim.z * blockDim.y * blockDim.x +
                threadIdx.z * blockDim.y * blockDim.x +
                threadIdx.y * blockDim.x +
                threadIdx.x;
}

// Returns true iff the point b is inside a's field of view
__device__ bool is_in_visual_field(glm::vec3 a, glm::vec3 b) {
    return glm::angle(glm::normalize(a), glm::normalize(b)) <= kVisualFieldAngle / 2 + epsilon;
}

__device__ void closest_neighbors(int*& points_indices, int& n_points,
        int global_index, int total_number,
        glm::vec3 *positions, glm::vec3 *directions,
        float radius) {
  n_points = 0;
  for(int j=0;j<total_number;++j){
    if(j != global_index){
      if(glm::distance(positions[j], positions[global_index]) < radius
         && is_in_visual_field(directions[global_index], positions[j] - positions[global_index])){
        points_indices[n_points] = j;
        n_points++;
      }
    }
  }
}


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

__device__ glm::vec3 stay_in_bounds(glm::vec3* positions, int cur_boid_index, int p=2){
    glm::vec3 new_vec(0,0,0);

    float x_max = 10.0;
    float y_max = 10.0;
    float z_max = 10.0;

    float x_min = -10.0;
    float y_min = -10.0;
    float z_min = -10.0;

    if(positions[cur_boid_index].x > x_max)
        new_vec.x -= pow(x_max - positions[cur_boid_index].x, p);
    if(positions[cur_boid_index].y > y_max)
        new_vec.y -= pow(y_max - positions[cur_boid_index].y, p);
    if(positions[cur_boid_index].z > z_max)
        new_vec.z -= pow(z_max - positions[cur_boid_index].z, p);

    if(positions[cur_boid_index].x < x_min)
        new_vec.x += pow(x_min - positions[cur_boid_index].x, p);
    if(positions[cur_boid_index].y < y_min)
        new_vec.y += pow(y_min - positions[cur_boid_index].y, p);
    if(positions[cur_boid_index].z < z_min)
        new_vec.z += pow(z_min - positions[cur_boid_index].z, p);

    return new_vec;
}

__device__ glm::vec3 calculate_direction(glm::vec3 from, glm::vec3 to){
    glm::vec3 goal(to.x, to.y, to.z);
    return goal - from;
}


__device__ glm::vec3 limit(glm::vec3 the_vec, float lim){
    float max_value = 0.0f;
    max_value = fmaxf(max_value, fmaxf(fabsf(the_vec.x),fmaxf(fabsf(the_vec.y), fabsf(the_vec.z))));
    if(max_value > lim)
        return the_vec*(lim/max_value);
    return the_vec;
}

__device__ glm::vec3 avoid_collisions(glm::vec3* positions, int* points_indices, int n_points, int cur_boid_index){
    glm::vec3 new_vec(.0f, .0f, .0f);
    for(int i=0; i< n_points; ++i){
        glm::vec3 offset = positions[points_indices[i]] - positions[cur_boid_index];
        new_vec += offset / glm::length(offset);
    }
    if (glm::length(new_vec) > 0)
        new_vec = -glm::normalize(new_vec);
    return new_vec;
}

// __global__ Functions

__global__ void GPU_update_vector(glm::vec3 *directions_output, glm::vec3 *directions_input, glm::vec3 *positions, int num_boids){
        int index = GPU_globalindex();
        if(index < num_boids)
        {
                if(fabsf(positions[index].x) > 15 || fabsf(positions[index].y) > 15 || fabsf(positions[index].z) > 15)
                    printf("[%d] WTFFF!!\n", index);
                glm::vec3 new_direction = glm::vec3(0,0,0);
                int n_points = 0;
                int* points_indices = new int[num_boids];

                closest_neighbors(points_indices, n_points, index,
                        num_boids, positions, directions_input, kRepulsionZoneRadius);

                // 1
                //new_direction += calculate_direction(positions[index], perceived_center(positions, points_indices, n_points, index)) * PERCEIVED_CENTER_WEIGHT;
                //glm::vec3 sum_vector = resultant(positions, points_indices, n_points, index);
                if (n_points) {
                    // since we have neighbors the repulsion behavior is applied
                    new_direction -= avoid_collisions(positions, points_indices, n_points, index);
                } else {
                    // if there aren't any neighbors in the repulsion zone
                    // we need to explore the orientation zone
                    closest_neighbors(points_indices, n_points, index, num_boids,
                            positions, directions_input, kOrientationZoneRadius);
                    glm::vec3 sum_vector = resultant(positions, points_indices, n_points, index);
                    glm::vec3 sum_direction =
                        resultant_direction(directions_input, points_indices, n_points, index);
                    if (n_points) {
                        new_direction +=
                            sum_vector * ATTRACTION_WEIGHT + sum_direction * ORIENTATION_WEIGHT;
                    }
                }

                new_direction += stay_in_bounds(positions, index, 2);//*8.0f;
                delete[] points_indices;
                new_direction = limit(new_direction, 1);
                directions_output[index] = directions_input[index] + new_direction*(ACCELERATION/100.0f);
                directions_output[index] = limit(directions_output[index], 1);
        }
}

__global__ void GPU_Update(glm::mat4 *modelMatrices, glm::vec3 *directions,
        glm::vec3 *updated_directions, glm::vec3 *positions,
        glm::vec3 *raxis, float *angles, int num_boids, float cT) {
        int i = GPU_globalindex();
        if(i < num_boids)
        {

                // float theta = 0.0;
                // glm::vec3 cr(0,0,0);
                // if(glm::length(directions[i]-updated_directions[i]) > epsilon)
                // {
                //         theta = glm::acos(glm::dot(glm::normalize(directions[i]),glm::normalize(updated_directions[i])));
                //         cr = glm::normalize(glm::cross(directions[i],updated_directions[i]));
                // }
                //
                // if(glm::length(raxis[i]) > epsilon)
                // {
                //         modelMatrices[i] = glm::rotate(modelMatrices[i], -angles[i], raxis[i]);
                // }
                //glm::vec3 new_direction = updated_directions[i]*(DELTA/(20.0f - SIMULATION_SPEED + 1));
                glm::vec3 new_direction = updated_directions[i]*DELTA;
                modelMatrices[i] = glm::translate(modelMatrices[i], new_direction);
                // if(glm::length(raxis[i]) > epsilon)
                // {
                //         modelMatrices[i] = glm::rotate(modelMatrices[i], angles[i], raxis[i]);
                // }
                //
                //
                // if(glm::length(cr) > epsilon)
                // {
                //         modelMatrices[i] = glm::rotate(modelMatrices[i], theta, cr);
                //         raxis[i] = glm::normalize(glm::cross(glm::vec3(0.0,0.0,0.25),updated_directions[i]));
                //         angles[i] = glm::acos(glm::dot(glm::normalize(updated_directions[i]),glm::normalize(glm::vec3(0.0,0.0,0.25))));;
                // }

                // TODO Remove this line, we perform cudaMemcpy later in the code
                directions[i] = updated_directions[i];
                positions[i] += new_direction;

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
