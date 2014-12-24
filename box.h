#pragma once

#include <vector>

#include <GL/glew.h>

#include <glm/glm.hpp>

class BoxMesh
{
public:
	BoxMesh(float dim_input, const char* filenameFragment, const char* filenameVertex);
	~BoxMesh();
	GLuint getProgramID();
	void draw(glm::mat4 VP);
	glm::mat4 getModelMatrix();
	// void setModelMatrix(glm::mat4 newModelMatrix);
	void setModelMatrix(glm::mat4 *newModelMatrix);
	void setColorTexture(const char* filename, const char* name);
	// void loadMesh(const char* filename);
private:
	GLuint vertexArrayID;
	GLuint programID;
	GLuint matrixID; 
	GLuint texture; 
	GLuint textureID;
	GLuint vertexbuffer;
	GLuint ebo;
	// glm::mat4 modelMatrix;
	glm::mat4 *modelMatrix;

	void VBO();
	float dim;
};

