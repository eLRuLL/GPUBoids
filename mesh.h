#pragma once

#include <vector>

#include <GL/glew.h>

#include <glm/glm.hpp>

class Mesh
{
public:
	Mesh(unsigned long long n_boids, const char* filenameFragment, const char* filenameVertex);
	~Mesh();
	GLuint getProgramID();
	void draw(glm::mat4 VP);
	glm::mat4 getModelMatrix();
	// void setModelMatrix(glm::mat4 newModelMatrix);
	void setModelMatrix(glm::mat4 *newModelMatrix);
	void setColorTexture(const char* filename, const char* name);
	void loadMesh(const char* filename);
private:
	GLuint vertexArrayID;
	GLuint programID;
	GLuint matrixID; 
	GLuint timeID; 
	GLuint texture; 
	GLuint textureID;
	GLuint orientationbuffer;
	GLuint positionbuffer;
	GLuint directionbuffer;
	GLuint vertexbuffer;
	GLuint uvbuffer;
	GLuint normalbuffer;
	// glm::mat4 modelMatrix;
	glm::mat4 *modelMatrix;
	// Read our .obj file
	std::vector<glm::vec3> vertices;
	std::vector<glm::vec2> uvs;
	std::vector<glm::vec3> normals;

	std::vector<glm::vec3> positions;
	std::vector<glm::mat4> orientations;
	std::vector<glm::vec3> directions;

	unsigned long long num_boids;

	void VBO();
};

