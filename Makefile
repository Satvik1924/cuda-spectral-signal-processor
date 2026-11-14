NVCC = nvcc
NVCC_FLAGS = --std=c++17 -O3 -Isrc -Wno-deprecated-gpu-targets
LDFLAGS = -lcufft
TARGET = bin/spectral_processor
SRCS = src/main.cu src/signal_processor.cu

.PHONY: all build clean

all: build

build: $(TARGET)

$(TARGET): $(SRCS) src/signal_processor.h
	mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

clean:
	rm -rf bin input_data output_data artifacts proof_of_execution.tar.gz
