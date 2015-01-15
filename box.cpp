#include <iostream>

#include "controls.h"
#include "box.h"
#include "objloader.h"
#include "shader.h"
#include "TextureManager.h"

#include "interop.h"


// Include GLM
#define GLM_FORCE_RADIANS

BoxMesh::BoxMesh(float dim_input, const char* filenameFragment, const char* filenameVertex)
{
	glGenVertexArrays(1, &vertexArrayID);
	glBindVertexArray(vertexArrayID);
	programID = LoadShaders(filenameFragment, filenameVertex);
	// Get a handle for our "MVP" uniform
	matrixID = glGetUniformLocation(programID, "VP");
	// Get a handle for our "LightPosition" uniform
	glUseProgram(programID);

	dim = dim_input/2.0f;

	VBO();


}

BoxMesh::~BoxMesh()
{
	// Cleanup VBO and shader
	glDeleteBuffers(1, &vertexbuffer);
	glDeleteProgram(programID);
	glDeleteTextures(1, &texture);
	glDeleteVertexArrays(1, &vertexArrayID);
}


void BoxMesh::VBO()
{
	// Load it into a VBO

  static const GLfloat g_vertex_buffer_data[] = {
  -dim,  -dim, -dim,
  dim,  -dim, -dim,
  dim,  dim, -dim,
  -dim,  dim, -dim,

  -dim,  -dim, dim,
  dim,  -dim, dim,
  dim,  dim, dim,
  -dim,  dim, dim,
    };

    glGenBuffers(1, &vertexbuffer);
    glBindBuffer(GL_ARRAY_BUFFER_ARB, vertexbuffer);
    glBufferData(GL_ARRAY_BUFFER_ARB, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

	GLuint elements[] = {
	    0, 1,	1, 2,	2, 3,	3, 0,
	    0, 4,	1, 5,	2, 6,	3, 7,
	    4, 5,	5, 6,	6, 7,	7, 4,
	};

	glGenBuffers(1, &ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STATIC_DRAW);

}

void BoxMesh::draw(glm::mat4 VP){			// MVP -> VP
	
	// glPolygonMode( GL_FRONT_AND_BACK, GL_LINE ); 

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
	glBindBuffer(GL_ARRAY_BUFFER_ARB, vertexbuffer);
	glVertexAttribPointer(
		0,                  // attribute
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		0,                  // stride
		(void*)0            // array buffer offset
		);

	// Draw the triangles !
	// glDrawArrays(GL_LINES, 0, 4);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, 0);
	glDisableVertexAttribArray(0);
	// glPolygonMode( GL_FRONT_AND_BACK, GL_FILL ); 
}

GLuint BoxMesh::getProgramID()
{
	return programID;
}

glm::mat4 BoxMesh::getModelMatrix()
{
	return *modelMatrix;
}
void BoxMesh::setModelMatrix(glm::mat4 *newModelMatrix)
{
	modelMatrix = newModelMatrix;
}


void BoxMesh::setColorTexture(const char* filename, const char* name)
{
	texture = TextureManager::Inst()->LoadTexture(filename);
	// Get a handle for our "myTextureSampler" uniform
	textureID = glGetUniformLocation(programID, name);
}

// void BoxMesh::loadMesh(const char* filename)
// {
// 	bool res = loadOBJ(filename, vertices, uvs, normals);

// 	VBO();

// }
