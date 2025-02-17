NVCC = nvcc
FILES = main.cu
TARGET = hello_cuda

STYLE ?= Google

$(TARGET): $(FILES)
	$(NVCC) -o $@ $(FILES)

clean:
	rm -f $(TARGET)

clang-format:
	clang-format -style=$(STYLE) -dump-config > .clang-format

format:
	find . -iregex '.*\.\(h\|c\|cpp\|cu\)$$' | xargs clang-format -i

