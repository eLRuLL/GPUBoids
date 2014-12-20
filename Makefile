all: main clean

main: cudacode
	g++ -std=c++11 *.cpp interop.o -o app -lGL -lGLU -lGLEW -lfreeimage -lglut -lglfw -L/usr/local/cuda/lib64 -lcudart

cudacode:
	# nvcc -c swarm.cu -o swarm.o
	nvcc -c interop.cu -o interop.o -lGL -lGLU -lGLEW -lfreeimage -lglut -gencode arch=compute_50,code=sm_50

clean:
	rm -f *.o
