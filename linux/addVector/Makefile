# Compiler
NVCC = nvcc

# Target executable
TARGET = addVector

# Source file
SRC = addVector.cu

# Compiler flags
CFLAGS = -O3

# Default target
all: $(TARGET)

# Compilation rule
$(TARGET): $(SRC)
	$(NVCC) $(SRC) -o $(TARGET) $(CFLAGS)

# Clean up
clean:
	rm -f $(TARGET)