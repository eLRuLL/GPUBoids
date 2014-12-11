all: main clean

main: cudacode
	g++ -std=c++11 -lGL -lglfw -lGLEW -lGLU -lglut -lfreeimage -L/opt/cuda/lib64 -lcudart *.cpp swarm.o -o app

cudacode:
	nvcc -c swarm.cu -o swarm.o

clean:
	rm -f *.o
