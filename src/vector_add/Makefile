NVCC = nvcc
NVCCFLAGS = -std=c++17 -arch=sm_86
CFLAGS =
FILES = src/main.cu
TARGET = vector_add

STYLE ?= Google

$(TARGET): $(FILES)
	$(NVCC) $(NVCCFLAGS) -o $@ $(CFLAGS) $(FILES)

clean:
	rm -f $(TARGET)

clang-format:
	clang-format -style=$(STYLE) -dump-config > .clang-format

format:
	find src -type f \( -iname '*.h' -o -iname '*.c' -o -iname '*.cpp' -o -iname '*.cu' \) -print0 | xargs -0 clang-format -i

