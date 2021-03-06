#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#define GLM_FORCE_RADIANS

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/norm.hpp>

#include "controls.h"
#include "main.h"
#include "mesh.h"
#include "shader.h"

using namespace glm;
using namespace std;

GLFWwindow* window;

int main( int argc, char *argv[])
{
	int num_boids = stoi(argv[1]);
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
	window = glfwCreateWindow(width, height,
                        "Fishschool Collective Behavior Simulation",
                        NULL /*glfwGetPrimaryMonitor()*/, NULL);
	if (window == NULL) {
		fprintf(stderr, "Failed to open GLFW window."
                        " If you have an Intel GPU, they are not 3.3 compatible."
                        " Try the 2.1 version of the tutorials.\n");
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
	uniform_real_distribution<float> fd_one(0.0,1.0);

	Mesh fish("Shaders/TransformVertexShader.vertexshader",
                  "Shaders/TextureFragmentShader.fragmentshader");
	fish.loadMesh("data/models/trout.obj");
	fish.setColorTexture("data/textures/jade.jpg", "myTextureSampler");
	vector<Mesh> swarm(num_boids,fish);
	vector<mat4> modelMatrices(num_boids,glm::mat4(1));
	vector<vec3> positions(num_boids,vec3(0.0,0.0,0.0));
	vector<vec3> directions(num_boids,vec3(0.0,0.0,0.25));
        for (auto &v : directions) {
                v[0] = fd_one(gen);
                v[1] = fd_one(gen);
        }

	vector<vec3> updated_directions(directions);
	vector<float> angle(num_boids,0.0);
	vector<vec3> raxis(num_boids,vec3(0,1,0));

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
    //positions[i] = vec3(0,0,0);
		modelMatrices[i] = translate(modelMatrices[i], positions[i]);
	}

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
                       &raxis[0], &angle[0], num_boids, float(currentTime));

                for (auto &fish : swarm) {
                        const mat4 MVP = ProjectionMatrix * ViewMatrix
                                * fish.getModelMatrix();
                        fish.draw(MVP);
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
