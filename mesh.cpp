#include <iostream>
#include <random>

#include "controls.h"
#include "mesh.h"
#include "objloader.h"
#include "shader.h"
#include "TextureManager.h"

#include "interop.h"


// Include GLM
#define GLM_FORCE_RADIANS

Mesh::Mesh(unsigned long long n_boids, const char* filenameFragment, const char* filenameVertex)
{
	num_boids = n_boids;
	glGenVertexArrays(1, &vertexArrayID);
	glBindVertexArray(vertexArrayID);
	programID = LoadShaders(filenameFragment, filenameVertex);
	// Get a handle for our "MVP" uniform
	matrixID = glGetUniformLocation(programID, "VP");

	// modelMatrix = glm::mat4(1);
	// modelMatrix = new glm::mat4(1);

	// Get a handle for our "LightPosition" uniform
	glUseProgram(programID);

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> fd(-8.0,8.0);

	positions.resize(num_boids);
	orientations.resize(num_boids, glm::mat4(1));
	directions.resize(num_boids, glm::vec3(0,0,1.0));

	for(unsigned long long i = 0; i<num_boids; i++){
		positions[i] = glm::vec3(fd(gen),fd(gen),fd(gen));
		// positions[i] = glm::vec3(0.0,0.0,0.0);
	}
}

Mesh::~Mesh()
{
	// Cleanup VBO and shader
	glDeleteBuffers(1, &vertexbuffer);
	glDeleteBuffers(1, &uvbuffer);
	glDeleteBuffers(1, &normalbuffer);
	glDeleteBuffers(1, &orientationbuffer);
	glDeleteBuffers(1, &positionbuffer);
	glDeleteBuffers(1, &directionbuffer);
	glDeleteProgram(programID);
	glDeleteTextures(1, &texture);
	glDeleteVertexArrays(1, &vertexArrayID);
	// delete modelMatrix;
	interop_cleanup();
}


void Mesh::VBO()
{
	// Load it into a VBO

	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices[0], GL_DYNAMIC_DRAW);

	glGenBuffers(1, &uvbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
	glBufferData(GL_ARRAY_BUFFER, uvs.size() * sizeof(glm::vec2), &uvs[0], GL_STATIC_DRAW);

	glGenBuffers(1, &normalbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
	glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(glm::vec3), &normals[0], GL_STATIC_DRAW);

	glGenBuffers(1, &orientationbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, orientationbuffer);
	glBufferData(GL_ARRAY_BUFFER, orientations.size() * sizeof(glm::mat4), &orientations[0], GL_DYNAMIC_DRAW);

	glGenBuffers(1, &positionbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, positionbuffer);
	glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(glm::vec3), &positions[0], GL_DYNAMIC_DRAW);

	glGenBuffers(1, &directionbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, directionbuffer);
	glBufferData(GL_ARRAY_BUFFER, directions.size() * sizeof(glm::vec3), &directions[0], GL_DYNAMIC_DRAW);

	interop_setup();
	interop_register_buffer(orientationbuffer, positionbuffer, directionbuffer);
	interop_map();
}

void Mesh::draw(glm::mat4 VP){			// MVP -> VP

	// Use our shader
	glUseProgram(programID);

	
	// Send our transformation to the currently bound shader,
	// in the "VP" uniform
	glUniformMatrix4fv(matrixID, 1, GL_FALSE, &VP[0][0]);
	
	// Bind our texture in Texture Unit 0
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);
	// Set our "myTextureSampler" sampler to user Texture Unit 0
	glUniform1i(textureID, 0);

	// 1rst attribute buffer : vertices
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glVertexAttribPointer(
		0,                  // attribute
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		0,                  // stride
		(void*)0            // array buffer offset
		);

	// 2nd attribute buffer : UVs
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
	glVertexAttribPointer(
		1,                                // attribute
		2,                                // size
		GL_FLOAT,                         // type
		GL_FALSE,                         // normalized?
		0,                                // stride
		(void*)0                          // array buffer offset
		);

	// 3rd attribute buffer : normals
	glEnableVertexAttribArray(2);
	glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
	glVertexAttribPointer(
		2,                                // attribute
		3,                                // size
		GL_FLOAT,                         // type
		GL_FALSE,                         // normalized?
		0,                                // stride
		(void*)0                          // array buffer offset
		);

	// 4th attribute buffer : positions
	glEnableVertexAttribArray(3);
	glBindBuffer(GL_ARRAY_BUFFER, positionbuffer);
	glVertexAttribPointer(
		3,                                // attribute
		3,                                // size
		GL_FLOAT,                         // type
		GL_FALSE,                         // normalized?
		0,                                // stride
		(void*)0                          // array buffer offset
		);

	// 4th, 5th, 6th, 7th attribute buffers : orientation
	glEnableVertexAttribArray(4);
	glEnableVertexAttribArray(5);
	glEnableVertexAttribArray(6);
	glEnableVertexAttribArray(7);
	glBindBuffer(GL_ARRAY_BUFFER, orientationbuffer);
	glVertexAttribPointer(
		4,                                // attribute
		4,                                // size
		GL_FLOAT,                         // type
		GL_FALSE,                         // normalized?
		sizeof(GLfloat) * 4 * 4,		  // stride
		(void*)0                          // array buffer offset
		);

	glVertexAttribPointer(
		5,                                // attribute
		4,                                // size
		GL_FLOAT,                         // type
		GL_FALSE,                         // normalized?
		sizeof(GLfloat) * 4 * 4,          // stride
		(void*)(sizeof(float) * 4)        // array buffer offset
		);

	glVertexAttribPointer(
		6,                                // attribute
		4,                                // size
		GL_FLOAT,                         // type
		GL_FALSE,                         // normalized?
		sizeof(GLfloat) * 4 * 4,          // stride
		(void*)(sizeof(float) * 8)        // array buffer offset
		);

	glVertexAttribPointer(
		7,                                // attribute
		4,                                // size
		GL_FLOAT,                         // type
		GL_FALSE,                         // normalized?
		sizeof(GLfloat) * 4 * 4,          // stride
		(void*)(sizeof(float) * 12)       // array buffer offset
		);

	// // 5th attribute buffer : directions
	// glEnableVertexAttribArray(4);
	// glBindBuffer(GL_ARRAY_BUFFER, directionbuffer);
	// glVertexAttribPointer(
	// 	4,                                // attribute
	// 	3,                                // size
	// 	GL_FLOAT,                         // type
	// 	GL_FALSE,                         // normalized?
	// 	0,                                // stride
	// 	(void*)0                          // array buffer offset
	// 	);

	glVertexAttribDivisor(0, 0);
	glVertexAttribDivisor(1, 0);
	glVertexAttribDivisor(2, 0);
	glVertexAttribDivisor(3, 1);
	glVertexAttribDivisor(4, 1);
	glVertexAttribDivisor(5, 1);
	glVertexAttribDivisor(6, 1);
	glVertexAttribDivisor(7, 1);

	// Draw the triangles !
	glDrawArraysInstanced(GL_TRIANGLES, 0, vertices.size(), num_boids);
	// glDrawArrays(GL_TRIANGLES, 0, vertices.size());

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glDisableVertexAttribArray(3);
	glDisableVertexAttribArray(4);
	glDisableVertexAttribArray(5);
	glDisableVertexAttribArray(6);
	glDisableVertexAttribArray(7);
}

GLuint Mesh::getProgramID()
{
	return programID;
}

glm::mat4 Mesh::getModelMatrix()
{
	return *modelMatrix;
}
void Mesh::setModelMatrix(glm::mat4 *newModelMatrix)
{
	modelMatrix = newModelMatrix;
}


void Mesh::setColorTexture(const char* filename, const char* name)
{
	texture = TextureManager::Inst()->LoadTexture(filename);
	// Get a handle for our "myTextureSampler" uniform
	textureID = glGetUniformLocation(programID, name);
}

void Mesh::loadMesh(const char* filename)
{
	bool res = loadOBJ(filename, vertices, uvs, normals);

	VBO();

}