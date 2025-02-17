NVCC = nvcc
CXX = clang++
CXXFLAGS =
TARGET = hello_cuda

STYLE ?= Google

all: main

main: main.o kernel.o
	$(CXX) $(CXXFLAGS) -o hello_cuda main.o kernel.o -L/usr/local/cuda/lib64 -lcudart

main.o: main.cpp kernel.h
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o -I/usr/local/cuda/include

kernel.o: kernel.cu kernel.h
	$(NVCC) -c kernel.cu -o kernel.o

clean:
	rm -f hello_cuda main.o kernel.o

clang-format:
	clang-format -style=$(STYLE) -dump-config > .clang-format

format:
	find . -iregex '.*\.\(h\|c\|cpp\|cu\)$$' | xargs clang-format -i

