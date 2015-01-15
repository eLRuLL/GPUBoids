all: main

main: cudacode
	# Este comando funciona para mi en MacOSX, modificarlo para LINUX
	/Developer/NVIDIA/CUDA-6.5/bin/nvcc -ccbin g++ -m64   -Xcompiler -arch -Xcompiler x86_64  -Xlinker -rpath -Xlinker /Developer/NVIDIA/CUDA-6.5/lib  -Xlinker -framework -Xlinker GLUT -gencode arch=compute_50,code=compute_50 -o app *.cpp interop.o -L/System/Library/Frameworks/OpenGL.framework/Libraries -lGL -lGLU -lglfw3 -lGLEW -lfreeimage

cudacode:
	# nvcc -c swarm.cu -o swarm.o
	nvcc -c -m64 interop.cu -o interop.o -lGL -lGLU -lGLEW -lfreeimage -lglut -gencode arch=compute_50,code=sm_50

clean:
	rm -f *.o
