# Compiler
NVCC = nvcc

# Target executable
TARGET = matMul

# Source file
SRC = matMul.cu

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