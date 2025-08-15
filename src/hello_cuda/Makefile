NVCC = nvcc
NVCCFLAGS = -std=c++17 -arch=sm_86
FILES = main.cu
TARGET = hello_cuda

STYLE ?= Google

$(TARGET): $(FILES)
	$(NVCC) $(NVCCFLAGS) -o $@ $(FILES)

clean:
	rm -f $(TARGET)

clang-format:
	clang-format -style=$(STYLE) -dump-config > .clang-format

format:
	find src -type f \( -iname '*.h' -o -iname '*.c' -o -iname '*.cpp' -o -iname '*.cu' \) -print0 | xargs -0 clang-format -i
