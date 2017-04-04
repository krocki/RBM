#OS detection
OS := $(shell uname)
ARCH := $(shell uname -m)
OS_VER := $(shell uname -r)
HOST := $(shell hostname | awk -F. '{print $$1}')

#detect number of cores
ifeq ($(OS),Linux)
	NPROCS := $(shell grep -c ^processor /proc/cpuinfo)
endif

ifeq ($(OS),Darwin)
	NPROCS := $(shell sysctl hw.ncpu | awk '{print $$2}')
endif

#OPTS
AVX=1
AVX2=0
OPENMP=0
CUDA=1
OPENGL=1

#CXX=g++
CXX=g++
INCLUDES=-I./src -I/usr/local/Cellar/openblas/0.2.18_2/include
LFLAGS=-lopenblas -L/usr/local/Cellar/openblas/0.2.18_2/lib
CFLAGS=-std=c++0x -w -O3 -mtune=native
CUDAFLAGS = -m64 -ccbin gcc -DUSE_CUBLAS -DUSE_CURAND --use_fast_math

CFLAGS := $(CFLAGS) -DDEFAULT_MAX_CORES=$(NPROCS)

ifeq ($(OS),Darwin)
	#empty
else
	LFLAGS := $(LFLAGS) -lpthread -lm
endif

ifeq ($(OPENMP), 1)
	CFLAGS := $(CFLAGS) -fopenmp
endif

ifeq ($(CUDA), 1)
	CFLAGS := $(CFLAGS) -DUSE_CUDA
endif

ifeq ($(OPENGL), 1)

	CFLAGS := $(CFLAGS) -DUSE_OPENGL

	ifeq ($(OS),Darwin)
		LFLAGS := $(LFLAGS) -framework GLUT -framework OpenGL -framework Cocoa
	else
		LFLAGS := $(LFLAGS) -lGL -lGLU -lglut
	endif

endif

ifeq ($(AVX2), 1)
	CFLAGS := $(CFLAGS) -mavx2
else
	ifeq ($(AVX), 1)
		CFLAGS := $(CFLAGS) -mavx
	else
		#empty
	endif
endif

all:
	$(CXX) ./src/mnist.cc $(INCLUDES) $(CFLAGS) $(LFLAGS) -o mnist
