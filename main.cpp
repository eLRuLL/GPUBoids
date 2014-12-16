

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <random>

#define GLM_FORCE_RADIANS
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/norm.hpp>

#include "shader.h"
#include "controls.h"
#include "mesh.h"

#include "main.h"

using namespace std;
using namespace glm;

GLFWwindow* window;

int main( int argc, char *argv[])
{

	int num_fishes = stoi(argv[1]);
	int p = stoi(argv[2]);
	const float epsilon = 1.0e-4;

	// Initialise GLFW
	if (!glfwInit())
	{
		fprintf(stderr, "Failed to initialize GLFW\n");
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Open a window and create its OpenGL context
	int width = 1280;	//1280 - 1920
	int height = 1024;	//1024 - 1080
	window = glfwCreateWindow(width, height, "Fishschool Collective Behavior Simulation", NULL/*glfwGetPrimaryMonitor()*/, NULL);
	if (window == NULL){
		fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	// glfwSetCursorPos(window, width / 2, height / 2);
	glfwSwapInterval(1);

	// Initialize GLEW
	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		return -1;
	}

	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

	// Dark blue background
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	// glClearColor(1.0f*88/255, 1.0f*129/255, 1.0f*243/255, 0.0f);

	// Enable depth test
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);
	// Cull triangles which normal is not towards the camera
	glEnable(GL_CULL_FACE);

	vec3 lightPos = vec3(0, 0, 0);

	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<float> fd(-10.0,10.0);

	Mesh fish("Shaders/TransformVertexShader.vertexshader", "Shaders/TextureFragmentShader.fragmentshader");
	fish.loadMesh("data/models/trout.obj");
	fish.setColorTexture("data/textures/jade.jpg", "myTextureSampler");
	vector<Mesh> swarm(num_fishes,fish);
	vector<mat4> modelMatrices(num_fishes,glm::mat4(1));
	vector<vec3> positions(num_fishes,vec3(0.0,0.0,0.0));
	vector<vec3> directions(num_fishes,vec3(0.0,0.0,0.25));
	vector<vec3> updated_directions(directions);
	vector<float> angle(num_fishes,0.0);
	vector<vec3> raxis(num_fishes,vec3(0,1,0));

	for(vector<Mesh>::iterator it = swarm.begin(); it != swarm.end(); it++)
	{
		int i = it-swarm.begin();
		it->setModelMatrix(&modelMatrices[i]);
	}

	for(vector<Mesh>::iterator it = swarm.begin(); it != swarm.end(); it++)
	{
		int i = it-swarm.begin();
		float cx, cy, cz;
		do
		{
			cx = fd(gen);
			cy = fd(gen);
			cz = fd(gen);
		}
		while((cx*cx + cy*cy + cz*cz > 10*10) || (cx*cx + cy*cy + cz*cz < 9.5*9.5));
		positions[i] = vec3(cx,cy,cz);
		modelMatrices[i] = translate(modelMatrices[i], positions[i]);
	}

	// Mesh shark("Shaders/TransformVertexShader.vertexshader", "Shaders/TextureFragmentShader.fragmentshader");
	// shark.loadMesh("data/models/shark.obj");
	// shark.setColorTexture("data/textures/sapphire.jpg", "myTextureSampler");
	// vector<Mesh> predators(p,shark);

/*	for(vector<Mesh>::iterator it = predators.begin(); it != predators.end(); it++)
	{
		int i = it-predators.begin();
		float cx, cy, cz;
		do
		{
			cx = fd(gen);
			cy = fd(gen);
			cz = fd(gen);
		}
		while((cx*cx + cy*cy + cz*cz > 10*10));
		vec3 cs(cx,cy,cz);
		vec3 cs(0,0,0);

		it->setModelMatrix(translate(it->getModelMatrix(), cs));
	}*/

	// mat4 ViewMatrix = translate(mat4(1.0f), vec3(0.0f,0.0f,-10.0f));
	// mat4 ViewMatrix2 = rotate(ViewMatrix, float(M_PI/2), vec3(1,0,0));
	// mat4 ViewMatrix3 = rotate(ViewMatrix, float(-M_PI/2), vec3(0,1,0));

	double lastTime = glfwGetTime();

	do
	{
		// Clear the screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// time between two frames
		double currentTime = glfwGetTime();
		double delta = currentTime - lastTime;
		lastTime = currentTime;

		// Compute the MVP matrix from keyboard and mouse input
		computeMatricesFromInputs();
		mat4 ProjectionMatrix = getProjectionMatrix();
		mat4 ViewMatrix = getViewMatrix();							

		update(&modelMatrices[0], &directions[0],
                       &updated_directions[0], &positions[0],
                       &raxis[0], &angle[0], num_fishes, float(currentTime));

		for(vector<Mesh>::iterator it = swarm.begin(); it != swarm.end(); it++)
		{
			mat4 MVP = ProjectionMatrix * ViewMatrix * it->getModelMatrix();
			it->draw(MVP);
		}

		// for(vector<Mesh>::iterator it = predators.begin(); it != predators.end(); it++)
		// {
		// 	// it->setModelMatrix(translate(it->getModelMatrix(), vec3(sin(currentTime),0.0,cos(currentTime))*float(delta)));
		// 	it->setModelMatrix(rotate(it->getModelMatrix(), 0.1f, vec3(0,1,0)));
			
		// 	glViewport (width/4, 0, width/2, height/2);
		// 	mat4 MVP = ProjectionMatrix * ViewMatrix * it->getModelMatrix();
		// 	it->draw(MVP);

		// 	glViewport (width/4, height/2, width/2, height/2);
		// 	MVP = ProjectionMatrix * ViewMatrix2 * it->getModelMatrix();
		// 	it->draw(MVP);

		// 	glViewport (width/2, 0, width/2, height/2);
		// 	MVP = ProjectionMatrix * ViewMatrix3 * it->getModelMatrix();
		// 	it->draw(MVP);
		// }

		// Swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();

	} // Check if the ESC key was pressed or the window was closed
	while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
	glfwWindowShouldClose(window) == 0);

	// Close OpenGL window and terminate GLFW
	glfwTerminate();

	return 0;
}

