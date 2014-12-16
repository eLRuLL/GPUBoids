#include <iostream>

#include "mesh.h"
#include "shader.h"
#include "TextureManager.h"
#include "objloader.h"
#include "controls.h"

// Include GLM
#define GLM_FORCE_RADIANS
#include <glm/gtc/matrix_transform.hpp>

Mesh::Mesh(const char* filenameFragment, const char* filenameVertex)
{
	glGenVertexArrays(1, &vertexArrayID);
	glBindVertexArray(vertexArrayID);
	programID = LoadShaders(filenameFragment, filenameVertex);
	// Get a handle for our "MVP" uniform
	matrixID = glGetUniformLocation(programID, "MVP");

	// modelMatrix = glm::mat4(1);
	// modelMatrix = new glm::mat4(1);

	// Get a handle for our "LightPosition" uniform
	glUseProgram(programID);
}

Mesh::~Mesh()
{
	// Cleanup VBO and shader
	glDeleteBuffers(1, &vertexbuffer);
	glDeleteBuffers(1, &uvbuffer);
	glDeleteBuffers(1, &normalbuffer);
	glDeleteProgram(programID);
	glDeleteTextures(1, &texture);
	glDeleteVertexArrays(1, &vertexArrayID);
	// delete modelMatrix;
}


void Mesh::VBO()
{
	// Load it into a VBO

	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices[0], GL_STATIC_DRAW);

	glGenBuffers(1, &uvbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
	glBufferData(GL_ARRAY_BUFFER, uvs.size() * sizeof(glm::vec2), &uvs[0], GL_STATIC_DRAW);

	glGenBuffers(1, &normalbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
	glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(glm::vec3), &normals[0], GL_STATIC_DRAW);

}

void Mesh::draw(glm::mat4 MVP){

	// Use our shader
	glUseProgram(programID);

	
	// Send our transformation to the currently bound shader,
	// in the "MVP" uniform
	glUniformMatrix4fv(matrixID, 1, GL_FALSE, &MVP[0][0]);

	
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

	// Draw the triangles !
	glDrawArrays(GL_TRIANGLES, 0, vertices.size());

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
}

GLuint Mesh::getProgramID()
{
	return programID;
}



// glm::mat4 Mesh::getModelMatrix()
// {
// 	return modelMatrix;
// }
// void Mesh::setModelMatrix(glm::mat4 newModelMatrix)
// {
// 	modelMatrix = newModelMatrix;
// }

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