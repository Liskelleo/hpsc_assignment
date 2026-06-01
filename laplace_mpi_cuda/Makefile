# # ============================================================
# # Makefile - 二维Laplace方程MPI+CUDA求解程序
# # ============================================================

# CXX = mpicxx
# NVCC = nvcc
# CXXFLAGS = -O3 -Wall -fopenmp
# CUDAFLAGS = -O3 -arch=sm_61 -std=c++14
# LDFLAGS = -lm -fopenmp

# TARGET = laplace_mpi_cuda
# SRCDIR = src
# OBJDIR = obj

# SOURCES = $(SRCDIR)/laplace_solver.cpp \
#           $(SRCDIR)/mpi_io.cpp

# CUDA_SOURCES = $(SRCDIR)/laplace_cuda.cu

# HEADERS = $(SRCDIR)/laplace_solver.h

# OBJECTS = $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
# CUDA_OBJECTS = $(CUDA_SOURCES:$(SRCDIR)/%.cu=$(OBJDIR)/%.o)

# all: $(TARGET)

# $(OBJDIR):
# 	mkdir -p $(OBJDIR)

# $(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(HEADERS) | $(OBJDIR)
# 	$(CXX) $(CXXFLAGS) -c $< -o $@

# $(OBJDIR)/%.o: $(SRCDIR)/%.cu $(HEADERS) | $(OBJDIR)
# 	$(NVCC) $(CUDAFLAGS) -c $< -o $@

# $(TARGET): $(OBJECTS) $(CUDA_OBJECTS)
# 	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# clean:
# 	rm -rf $(OBJDIR) $(TARGET)

# run: $(TARGET)
# 	mpirun -np 2 ./$(TARGET) 256 256 1e-6 50000 sor

# # 不同迭代方法的测试
# test_jacobi: $(TARGET)
# 	mpirun -np 2 ./$(TARGET) 256 256 1e-6 50000 jacobi

# test_gs: $(TARGET)
# 	mpirun -np 2 ./$(TARGET) 256 256 1e-6 50000 gs

# test_sor: $(TARGET)
# 	mpirun -np 2 ./$(TARGET) 256 256 1e-6 50000 sor

# # CUDA版本测试
# test_cuda: $(TARGET)
# 	mpirun -np 2 ./$(TARGET) 512 512 1e-6 50000 sor 1

# # 异步Stream测试（选做）
# test_async: $(TARGET)
# 	mpirun -np 2 ./$(TARGET) 512 512 1e-6 50000 sor 1 1

# # 弱扩展性测试
# weak_scaling: $(TARGET)
# 	@echo "=== Weak Scaling测试 ==="
# 	@echo "进程数: 1 | 网格: 256×256"
# 	mpirun -np 1 ./$(TARGET) 256 256 1e-6 50000 sor
# 	@echo "进程数: 2 | 网格: 362×362"
# 	mpirun -np 2 ./$(TARGET) 362 362 1e-6 50000 sor
# 	@echo "进程数: 4 | 网格: 512×512"
# 	mpirun -np 4 ./$(TARGET) 512 512 1e-6 50000 sor
# 	@echo "进程数: 8 | 网格: 724×724"
# 	mpirun -np 8 ./$(TARGET) 724 724 1e-6 50000 sor

# # 强扩展性测试
# strong_scaling: $(TARGET)
# 	@echo "=== Strong Scaling测试 (512×512) ==="
# 	@for np in 1 2 4 8; do \
# 	    echo -n "进程数: $$np | "; \
# 	    mpirun -np $$np ./$(TARGET) 512 512 1e-6 50000 sor 2>/dev/null | grep "计算时间"; \
# 	done

# # Profiling（需要nvprof或nsys）
# profile: $(TARGET)
# 	mpirun -np 2 nvprof ./$(TARGET) 1024 1024 1e-6 10000 sor 1

# .PHONY: all clean run test_jacobi test_gs test_sor test_cuda test_async \
#         weak_scaling strong_scaling profile

# # Profiling目标
# profile: $(TARGET)
# 	chmod +x profile.sh
# 	./profile.sh

# # 分析nsys报告
# analyze_nsys:
# 	@echo "分析nsys报告..."
# 	@if [ -f laplace_profile_small.nsys-rep ]; then \
# 	    nsys stats laplace_profile_small.nsys-rep; \
# 	fi

# # 生成CUDA kernel占用率报告
# occupancy: $(TARGET)
# 	mpirun -np 1 nvprof --metrics achieved_occupancy \
# 	    ./$(TARGET) 256 256 1e-6 1000 sor 1

# # 内存带宽分析
# bandwidth: $(TARGET)
# 	mpirun -np 1 nvprof --metrics dram_read_throughput,dram_write_throughput \
# 	    ./$(TARGET) 256 256 1e-6 1000 sor 1

# ============================================================
# Makefile - 二维Laplace方程MPI+CUDA求解程序 (WSL2 优化版)
# ============================================================

CXX = mpicxx
NVCC = nvcc

# 普通C++编译器：包含CUDA头文件
CXXFLAGS = -O3 -Wall -fopenmp -I/usr/local/cuda/include

# 关键修复：给NVCC编译器补上MPI的头文件路径
CUDAFLAGS = -O3 -arch=sm_75 -std=c++17 -I/usr/lib/x86_64-linux-gnu/openmpi/include

# 链接器：显式链接CUDA运行时库
LDFLAGS = -lm -fopenmp -L/usr/local/cuda/lib64 -lcudart

TARGET = laplace_mpi_cuda
SRCDIR = src
OBJDIR = obj

# 基础 C++ 源码
SOURCES = $(SRCDIR)/laplace_solver.cpp \
          $(SRCDIR)/mpi_io.cpp

# 关键修复 4：把所有 CUDA 源码文件全部抓进来
CUDA_SOURCES = $(SRCDIR)/laplace_cuda.cu \
               $(SRCDIR)/stream_async.cu \
               $(SRCDIR)/main_async.cu

# 包含所有的头文件依赖
HEADERS = $(SRCDIR)/laplace_solver.h \
          $(SRCDIR)/stream_async.h

OBJECTS = $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
CUDA_OBJECTS = $(CUDA_SOURCES:$(SRCDIR)/%.cu=$(OBJDIR)/%.o)

all: $(TARGET)

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(HEADERS) | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu $(HEADERS) | $(OBJDIR)
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

$(TARGET): $(OBJECTS) $(CUDA_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -rf $(OBJDIR) $(TARGET)

# 下面你原本自带的自动化测试目标全部保持不变
run: $(TARGET)
	mpirun -np 2 ./$(TARGET) 256 256 1e-6 50000 sor

test_jacobi: $(TARGET)
	mpirun -np 2 ./$(TARGET) 256 256 1e-6 50000 jacobi

test_gs: $(TARGET)
	mpirun -np 2 ./$(TARGET) 256 256 1e-6 50000 gs

test_sor: $(TARGET)
	mpirun -np 2 ./$(TARGET) 256 256 1e-6 50000 sor

test_cuda: $(TARGET)
	mpirun -np 2 ./$(TARGET) 512 512 1e-6 50000 sor 1

test_async: $(TARGET)
	mpirun -np 2 ./$(TARGET) 512 512 1e-6 50000 sor 1 1

weak_scaling: $(TARGET)
	@echo "=== Weak Scaling测试 ==="
	@echo "进程数: 1 | 网格: 256×256"
	mpirun -np 1 ./$(TARGET) 256 256 1e-6 50000 sor
	@echo "进程数: 2 | 网格: 362×362"
	mpirun -np 2 ./$(TARGET) 362 362 1e-6 50000 sor
	@echo "进程数: 4 | 网格: 512×512"
	mpirun -np 4 ./$(TARGET) 512 512 1e-6 50000 sor
	@echo "进程数: 8 | 网格: 724×724"
	mpirun -np 8 ./$(TARGET) 724 724 1e-6 50000 sor

strong_scaling: $(TARGET)
	@echo "=== Strong Scaling测试 (512×512) ==="
	@for np in 1 2 4 8; do \
	    echo -n "进程数: $$np | "; \
	    mpirun -np $$np ./$(TARGET) 512 512 1e-6 50000 sor 2>/dev/null | grep "计算时间"; \
	done

profile: $(TARGET)
	chmod +x profile.sh
	./profile.sh

analyze_nsys:
	@echo "分析nsys报告..."
	@if [ -f laplace_profile_small.nsys-rep ]; then \
	    nsys stats laplace_profile_small.nsys-rep; \
	fi

occupancy: $(TARGET)
	mpirun -np 1 nvprof --metrics achieved_occupancy ./$(TARGET) 256 256 1e-6 1000 sor 1

bandwidth: $(TARGET)
	mpirun -np 1 nvprof --metrics dram_read_throughput,dram_write_throughput ./$(TARGET) 256 256 1e-6 1000 sor 1

.PHONY: all clean run test_jacobi test_gs test_sor test_cuda test_async \
        weak_scaling strong_scaling profile analyze_nsys occupancy bandwidth