#define GL_GLEXT_PROTOTYPES
#include <GL/glut.h>
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/norm.hpp>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <vector>

#include <stdio.h>

// cudaGraphicsResource *resource;
cudaGraphicsResource *o_resource;
cudaGraphicsResource *p_resource;
// float3 *devPtr;
glm::mat4 *o_devPtr;
glm::vec3 *p_devPtr;



// __global__ void kernel( float3 *ptr ) {
__global__ void kernel( glm::mat4 *ptr, glm::vec3 *pos, float cT, float delta) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int index = x + y * blockDim.x * gridDim.x;
	// now calculate the value at that position
	// glm::vec3 v(0,0,0.015);
	glm::vec3 direction(sinf(cT),0,cosf(cT));
	glm::vec3 v = glm::cross(glm::vec3(0,0,1),glm::normalize(direction));
	float sine = glm::length(v);
	float cosine = glm::dot(glm::vec3(0,0,1),glm::normalize(direction));
	glm::mat3 v_x = glm::mat3(0.0f,v[2],-v[1],
								-v[2],0.0f,v[0],
								v[1],-v[0],0.0f);
	glm::mat3 R = glm::mat3(1) + v_x + (v_x)*(v_x)*(1-cosine)/(sine*sine);

	ptr[index] = glm::mat4(R);
	pos[index] += glm::normalize(direction)*delta;
}


void interop_setup() {
	cudaDeviceProp prop;
	int dev;
	memset( &prop, 0, sizeof( cudaDeviceProp ) );
	prop.major = 1;
	prop.minor = 0;
	cudaChooseDevice( &dev, &prop );
	cudaGLSetGLDevice( dev );	// dev = 0
}

void interop_register_buffer(GLuint& o_buffer, GLuint& p_buffer){
	cudaGraphicsGLRegisterBuffer( &o_resource,o_buffer,cudaGraphicsMapFlagsNone);
	cudaGraphicsGLRegisterBuffer( &p_resource,p_buffer,cudaGraphicsMapFlagsNone);
}

void interop_map() {
	size_t o_size;
	cudaGraphicsMapResources( 1, &o_resource, NULL );
	cudaGraphicsResourceGetMappedPointer( (void**)&o_devPtr,&o_size,o_resource ) ;

	size_t p_size;
	cudaGraphicsMapResources( 1, &p_resource, NULL );
	cudaGraphicsResourceGetMappedPointer( (void**)&p_devPtr,&p_size,p_resource ) ;
}

void interop_run(int num_boids, float cT, float delta) {
	dim3 grids(num_boids,1);
	dim3 threads(1,1);
	kernel<<<grids,threads>>>( o_devPtr, p_devPtr,cT, delta);
	cudaGraphicsUnmapResources( 1, &o_resource, NULL );
	cudaGraphicsUnmapResources( 1, &p_resource, NULL );
}

void interop_cleanup(){
	cudaGraphicsUnregisterResource( o_resource );
	cudaGraphicsUnregisterResource( p_resource );
}