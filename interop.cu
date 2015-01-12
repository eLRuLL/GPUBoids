#define GL_GLEXT_PROTOTYPES
#include <GL/glut.h>
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtx/vector_angle.hpp>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <vector>

#include <stdio.h>

const float kRepulsionZoneRadius = 0.5;
const float kOrientationZoneRadius = 3.0;
const float kVisualFieldAngle = 3.14159 * 120.0 / 180.0;

#define REPULSION_WEIGHT -1.0f
#define ATTRACTION_WEIGHT 5.0f
#define ORIENTATION_WEIGHT 10.0f

#define ACCELERATION 2.0f

// const float epsilon = 1.0e-4;

cudaGraphicsResource *o_resource;
cudaGraphicsResource *p_resource;
cudaGraphicsResource *d_resource;
glm::mat4 *o_devPtr;
glm::vec3 *p_devPtr;
glm::vec3 *d_devPtr;
glm::vec3 *newd_devPtr;

// __device__ Functions



__device__ int GPU_globalindex() {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int index = x + y * blockDim.x * gridDim.x;
    return index;
}

// Returns true iff the point b is inside a's field of view
__device__ bool is_in_visual_field(glm::vec3 a, glm::vec3 b) {
    return glm::angle(glm::normalize(a), glm::normalize(b)) <= kVisualFieldAngle / 2;
}


__device__ void closest_neighbors(int*& points_indices, int& n_points, int global_index, int total_number, glm::vec3 *positions, glm::vec3 *directions, float radius) {
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
        if (glm::length(positions[points_indices[j]] - positions[cur_boid_index]) > 0)
            new_vec += glm::normalize(positions[points_indices[j]] - positions[cur_boid_index]);
    return new_vec;
}

__device__ glm::vec3 resultant_direction(glm::vec3* directions, int* points_indices, int n_points, int cur_boid_index){
    glm::vec3 new_vec(0,0,0);
    for(int j=0;j<n_points;++j)
        if (glm::length(directions[points_indices[j]]) > 0)
            new_vec += glm::normalize(directions[points_indices[j]]);
    return new_vec;
}

__device__ glm::vec3 perceived_center(glm::vec3* positions, int* points_indices, int n_points, int cur_boid_index){
    glm::vec3 new_vec(0,0,0);
    for(int i=0; i<n_points; ++i){
      new_vec += positions[points_indices[i]];
    }
    return new_vec*(1.0f/n_points);
}

__device__ glm::vec3 stay_in_bounds(glm::vec3* positions, int cur_boid_index){
    glm::vec3 new_vec(0,0,0);

    float x_max = 10.0;
    float y_max = 10.0;
    float z_max = 10.0;

    float x_min = -10.0;
    float y_min = -10.0;
    float z_min = -10.0;

    // if(positions[cur_boid_index].x > x_max - kRepulsionZoneRadius)
    //     new_vec.z -= positions[cur_boid_index].z + z_max;
    // if(positions[cur_boid_index].y > y_max - kRepulsionZoneRadius)
    //     new_vec.y -= positions[cur_boid_index].y + y_max;
    // if(positions[cur_boid_index].z > z_max - kRepulsionZoneRadius)
    //     new_vec.x = positions[cur_boid_index].x + x_max;

    // if(positions[cur_boid_index].x < x_min + kRepulsionZoneRadius)
    //     new_vec.z = positions[cur_boid_index].z + z_max;
    // if(positions[cur_boid_index].y < y_min + kRepulsionZoneRadius)
    //     new_vec.y = positions[cur_boid_index].y + y_max;
    // if(positions[cur_boid_index].z < z_min + kRepulsionZoneRadius)
    //     new_vec.x -= positions[cur_boid_index].x + x_max;

    if(positions[cur_boid_index].z > z_max - kRepulsionZoneRadius)
        new_vec.z -= positions[cur_boid_index].z + z_max;
    if(positions[cur_boid_index].y > y_max - kRepulsionZoneRadius)
        new_vec.y -= positions[cur_boid_index].y + y_max;
    if(positions[cur_boid_index].x > x_max - kRepulsionZoneRadius)
        new_vec.x -= positions[cur_boid_index].x + x_max;

    if(positions[cur_boid_index].z < z_min + kRepulsionZoneRadius)
        new_vec.z += positions[cur_boid_index].z + z_max;
    if(positions[cur_boid_index].y < y_min + kRepulsionZoneRadius)
        new_vec.y += positions[cur_boid_index].y + y_max;
    if(positions[cur_boid_index].x < x_min + kRepulsionZoneRadius)
        new_vec.x += positions[cur_boid_index].x + x_max;

    if(glm::length(new_vec) > 0)
        new_vec = glm::normalize(new_vec);

    return new_vec;
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
        if (glm::length(offset) > 0)
            new_vec -= glm::normalize(offset);
    }
    return new_vec;
}


__device__ void GPU_Update_Direction(glm::vec3 *directions_output, glm::vec3 *directions_input, glm::vec3 *positions, int index,int num_boids){
            // glm::vec3 new_direction = glm::vec3(0,0,0);
            // int n_points = 0;
            // int* points_indices = new int[num_boids];

            // closest_neighbors(points_indices, n_points, index,
            //         num_boids, positions, directions_input, kRepulsionZoneRadius);

            // if (n_points) {
            //     // since we have neighbors the repulsion behavior is applied
            //     new_direction = avoid_collisions(positions, points_indices, n_points, index);
            // } else {
            //     // if there aren't any neighbors in the repulsion zone
            //     // we need to explore the orientation zone
            //     closest_neighbors(points_indices, n_points, index, num_boids,
            //             positions, directions_input, kOrientationZoneRadius);
            //     glm::vec3 sum_vector = resultant(positions, points_indices, n_points, index);
            //     glm::vec3 sum_direction =
            //         resultant_direction(directions_input, points_indices, n_points, index);
            //     if (n_points) {
            //         new_direction =
            //             sum_vector * ATTRACTION_WEIGHT + sum_direction * ORIENTATION_WEIGHT;
            //     }
            // }

            // delete[] points_indices;
            // if(glm::length(new_direction) > 0)
            //     directions_output[index] = glm::normalize(new_direction)*(ACCELERATION/500.0f);
            // else
            

            // directions_output[index] = directions_input[index];
                // directions_output[index] = stay_in_bounds(positions, index);
            

            glm::vec3 sb = stay_in_bounds(positions, index);
            
            
                            glm::vec3 new_direction = glm::vec3(0,0,0);
            int n_points = 0;
            int* points_indices = new int[num_boids];

            closest_neighbors(points_indices, n_points, index,
                    num_boids, positions, directions_input, kRepulsionZoneRadius);

            if (n_points) {
                // since we have neighbors the repulsion behavior is applied
                new_direction = avoid_collisions(positions, points_indices, n_points, index);
            }
            // } else {
            //     // if there aren't any neighbors in the repulsion zone
            //     // we need to explore the orientation zone
            //     closest_neighbors(points_indices, n_points, index, num_boids,
            //             positions, directions_input, kOrientationZoneRadius);
            //     glm::vec3 sum_vector = resultant(positions, points_indices, n_points, index);
            //     glm::vec3 sum_direction =
            //         resultant_direction(directions_input, points_indices, n_points, index);
            //     if (n_points) {
            //         new_direction =
            //             sum_vector /** ATTRACTION_WEIGHT*/ + sum_direction /** ORIENTATION_WEIGHT*/;
            //     }
            // }

            delete[] points_indices;
            if(glm::length(new_direction) > 0)
                directions_output[index] = glm::normalize(new_direction);
                else
                    directions_output[index] = directions_input[index];

            if(glm::length(sb) > 0)
                directions_output[index] += sb;

            directions_input[index] = directions_output[index];


}

__global__ void GPU_Update( glm::mat4 *orient, glm::vec3 *pos, glm::vec3 *dir_input, glm::vec3 *dir_output, int num_boids, float delta) {
    // map from threadIdx/BlockIdx to pixel position
    int index = GPU_globalindex();
    // now calculate the value at that position
    GPU_Update_Direction(dir_output, dir_input, pos, index, num_boids);

    if(glm::length(dir_output[index]) > 0)
    {
        glm::vec3 v = glm::cross(glm::vec3(0,0,1),glm::normalize(dir_output[index]));
        float sine = glm::length(v);
        // if (sine != 0.0)
        // {
        //     float cosine = glm::dot(glm::vec3(0,0,1),glm::normalize(dir_output[index]));
        //     glm::mat3 v_x = glm::mat3(0.0f,v[2],-v[1],
        //                                 -v[2],0.0f,v[0],
        //                                 v[1],-v[0],0.0f);

        //     // if(glm::asin(sine) > kVisualFieldAngle * delta /2 )
        //     // {
        //     //     sine = glm::sin(kVisualFieldAngle * delta /2);
        //     //     cosine = glm::cos(kVisualFieldAngle * delta/2);

        //     //     glm::mat4 rotationMat(1); // Creates a identity matrix
        //     //     rotationMat = glm::rotate(rotationMat, kVisualFieldAngle*delta/2, v);
        //     //     dir_output[index] = glm::vec3(rotationMat * glm::vec4(dir_output[index], 1.0));
        //     // }
        //     // else if(glm::asin(sine) < - kVisualFieldAngle * delta/2 )
        //     // {
        //     //     sine = glm::sin(-kVisualFieldAngle * delta/2);
        //     //     cosine = glm::cos(-kVisualFieldAngle * delta/2);
        //     //     glm::mat4 rotationMat(1); // Creates a identity matrix
        //     //     rotationMat = glm::rotate(rotationMat, -kVisualFieldAngle*delta/2, v);
        //     //     dir_output[index] = glm::vec3(rotationMat * glm::vec4(dir_output[index], 1.0));
        //     // }

        //     glm::mat3 R = glm::mat3(1) + v_x + (v_x)*(v_x)*(1-cosine)/(sine*sine);
        
        //     orient[index] = glm::mat4(R);
        // }
        pos[index] += glm::normalize(dir_output[index])*delta;
    }
}


void interop_setup() {
    cudaDeviceProp prop;
    int dev;
    memset( &prop, 0, sizeof( cudaDeviceProp ) );
    prop.major = 1;
    prop.minor = 0;
    cudaChooseDevice( &dev, &prop );
    cudaGLSetGLDevice( dev );   // dev = 0
}

void interop_register_buffer(GLuint& o_buffer, GLuint& p_buffer, GLuint& d_buffer){
    cudaGraphicsGLRegisterBuffer( &o_resource,o_buffer,cudaGraphicsMapFlagsNone);
    cudaGraphicsGLRegisterBuffer( &p_resource,p_buffer,cudaGraphicsMapFlagsNone);
    cudaGraphicsGLRegisterBuffer( &d_resource,d_buffer,cudaGraphicsMapFlagsNone);
}

void interop_map() {
    size_t o_size;
    cudaGraphicsMapResources( 1, &o_resource, NULL );
    cudaGraphicsResourceGetMappedPointer( (void**)&o_devPtr,&o_size,o_resource ) ;

    size_t p_size;
    cudaGraphicsMapResources( 1, &p_resource, NULL );
    cudaGraphicsResourceGetMappedPointer( (void**)&p_devPtr,&p_size,p_resource ) ;

    size_t d_size;
    cudaGraphicsMapResources( 1, &d_resource, NULL );
    cudaGraphicsResourceGetMappedPointer( (void**)&d_devPtr,&d_size,d_resource ) ;


}

void interop_run(int num_boids, float delta) {

    cudaMalloc(&newd_devPtr, num_boids * sizeof(glm::vec3));


    dim3 grids(num_boids,1);
    dim3 threads(1,1);
    GPU_Update<<<grids,threads>>>( o_devPtr, p_devPtr, d_devPtr, newd_devPtr, num_boids, delta);
    cudaGraphicsUnmapResources( 1, &o_resource, NULL );
    cudaGraphicsUnmapResources( 1, &p_resource, NULL );
    cudaGraphicsUnmapResources( 1, &d_resource, NULL );

    cudaFree(newd_devPtr);

}

void interop_cleanup(){
    cudaGraphicsUnregisterResource( o_resource );
    cudaGraphicsUnregisterResource( p_resource );
    cudaGraphicsUnregisterResource( d_resource );
}