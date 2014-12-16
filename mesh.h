#pragma once

#include <GL/glew.h>
#include <vector>
#include <glm/glm.hpp>

class Mesh
{
public:
	Mesh(const char* filenameFragment, const char* filenameVertex);
	~Mesh();
	GLuint getProgramID();
	void draw(glm::mat4 MVP);
	glm::mat4 getModelMatrix();
	// void setModelMatrix(glm::mat4 newModelMatrix);
	void setModelMatrix(glm::mat4 *newModelMatrix);
	void setColorTexture(const char* filename, const char* name);
	void loadMesh(const char* filename);
private:
	GLuint vertexArrayID;
	GLuint programID;
	GLuint matrixID; 
	GLuint texture; 
	GLuint textureID;
	GLuint vertexbuffer;
	GLuint uvbuffer;
	GLuint normalbuffer;
	// glm::mat4 modelMatrix;
	glm::mat4 *modelMatrix;
	// Read our .obj file
	std::vector<glm::vec3> vertices;
	std::vector<glm::vec2> uvs;
	std::vector<glm::vec3> normals;



	void VBO();
};

