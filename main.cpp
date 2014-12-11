
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

	int n = stoi(argv[1]);

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
	int width = 1280;
	int height = 1024;
	window = glfwCreateWindow(width, height, "Solar System", NULL/*glfwGetPrimaryMonitor()*/, NULL);
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
	glClearColor(0.0f, 0.0f, 0.1f, 0.0f);
	// glClearColor(1.0f*88/255, 1.0f*129/255, 1.0f*243/255, 0.0f);

	// Enable depth test
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);
	// Cull triangles which normal is not towards the camera
	glEnable(GL_CULL_FACE);

	vec3 lightPos = vec3(0, 0, 0);


	Mesh fish("Shaders/TransformVertexShader.vertexshader", "Shaders/TextureFragmentShader.fragmentshader");
	fish.loadMesh("data/models/trout4.obj");
	fish.setColorTexture("data/textures/colorMoon.png", "myTextureSampler");
	vector<Mesh> swarm(n,fish);
	vector<vec3> c(n,vec3(0.0,0.0,0.0));
	vector<vec3> d(n,vec3(0.0,0.0,0.25));
	vector<vec3> dj(d);
	vector<float> w(n,0.0);

	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<float> fd(-10.0,10.0);

	for(vector<Mesh>::iterator it = swarm.begin(); it != swarm.end(); it++)
	{
		int i = it-swarm.begin();
		c[i] = vec3(fd(gen),fd(gen),fd(gen));

		it->setModelMatrix(translate(it->getModelMatrix(), c[i] ));
	}

	cout << "READY!!!" << endl;

	float speed = 10.0f;
	mat4 ViewMatrix = mat4(1.0f);
	ViewMatrix = translate(ViewMatrix, vec3(0.0f,0.0f,-5.0f));
	ViewMatrix = rotate(ViewMatrix, float(M_PI/2), vec3(1,0,0));

	double lastTime = glfwGetTime();
	do
	{
		// Clear the screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		// glViewport (320, 28, 1280, 1024);

		// time between two frames
		double currentTime = glfwGetTime();
		double delta = currentTime - lastTime;
		lastTime = currentTime;

		// Compute the MVP matrix from keyboard and mouse input
		computeMatricesFromInputs();
		mat4 ProjectionMatrix = getProjectionMatrix();
		// mat4 ViewMatrix = getViewMatrix();

		update(&dj[0],&c[0],&w[0], n);

		for(vector<Mesh>::iterator it = swarm.begin(); it != swarm.end(); it++)
		{
			int i = it-swarm.begin();

			// cout << "d[" << i << "] = " << d[i].x << "\t;" << d[i].y << "\t;" << d[i].z << endl;
			// cout << "c[" << i << "] = " << c[i].x << "\t;" << c[i].y << "\t;" << c[i].z << endl;
			// cout << endl;

			// vec3 dj(0.25,0.0,0.25*(currentTime>1.0));



			float theta = acos(dot(normalize(d[i]),normalize(dj[i])));
			d[i] = dj[i];

			it->setModelMatrix(translate(it->getModelMatrix(), d[i]*float(delta)));
			it->setModelMatrix(rotate(it->getModelMatrix(), theta*float(delta), vec3(0, 1, 0)));		// Mejorar el Giro

			c[i] += d[i]*float(delta);
			mat4 MVP = ProjectionMatrix * ViewMatrix * it->getModelMatrix();
			it->draw(MVP);
		}

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

