all: main clean

main: cudacode
	g++ -std=c++11 *.cpp swarm.o -o app -lGL -lGLU -lGLEW -lfreeimage -lglut -lglfw -L/usr/local/cuda/lib64 -lcudart

cudacode:
	nvcc -c swarm.cu -o swarm.o

clean:
	rm -f *.o
