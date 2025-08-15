NVCC = nvcc
NVCCFLAGS = -std=c++17 -arch=sm_86
CXX = clang++
CXXFLAGS =
TARGET = hello_cuda

STYLE ?= Google

all: main

main: main.o kernel.o
	$(CXX) $(CXXFLAGS) -o hello_cuda main.o kernel.o -L/usr/local/cuda/lib64 -lcudart

main.o: src/main.cpp src/kernel.h
	$(CXX) $(CXXFLAGS) -c src/main.cpp -o main.o -I/usr/local/cuda/include

kernel.o: src/kernel.cu src/kernel.h
	$(NVCC) $(NVCCFLAGS) -c src/kernel.cu -o kernel.o

clean:
	rm -f hello_cuda main.o kernel.o

clang-format:
	clang-format -style=$(STYLE) -dump-config > .clang-format

format:
	find src -type f \( -iname '*.h' -o -iname '*.c' -o -iname '*.cpp' -o -iname '*.cu' \) -print0 | xargs -0 clang-format -i

