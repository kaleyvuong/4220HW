CC     = gcc
NVCC   = nvcc
CFLAGS = -O2

NVFLAGS = -O2 -arch=sm_80 -lcurand --allow-unsupported-compiler -L/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/math_libs/12.8/targets/x86_64-linux/lib

all: cpu_baseline gpu_mc

cpu_baseline: monte_carlo_cpu.c
	$(CC) $(CFLAGS) -o cpu_baseline monte_carlo_cpu.c -lm

gpu_mc: monte_carlo_gpu.cu
	$(NVCC) $(NVFLAGS) -o gpu_mc monte_carlo_gpu.cu

clean:
	rm -f cpu_baseline gpu_mc
