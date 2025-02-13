CC = nvcc
CFLAGS =
FILES = main.cu
TARGET = vector_add

STYLE ?= Google

$(TARGET): $(FILES)
	$(CC) -o $@ $(CFLAGS) $(FILES)

clean:
	rm -f $(TARGET)

clang-format:
	clang-format -style=$(STYLE) -dump-config > .clang-format

format:
	find . -iregex '.*\.\(h\|c\|cpp\|cu\)$$' | xargs clang-format -i

