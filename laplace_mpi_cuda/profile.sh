#!/bin/bash
# ============================================================
# Profiling脚本 - 使用nvprof/nsys分析CUDA kernel性能
# ============================================================

cd ~/laplace_mpi_cuda

# 编译profiling版本
make clean
make CXXFLAGS="-O2 -g -Wall -fopenmp" CUDAFLAGS="-O2 -g -arch=sm_61 -lineinfo"

echo "=========================================="
echo "CUDA Profiling - Laplace Solver"
echo "=========================================="

# 使用nsys（推荐，CUDA 11+）
if command -v nsys &> /dev/null; then
    echo ""
    echo "=== nsys profiling ==="
    
    # 小网格profiling
    mpirun -np 1 nsys profile -o laplace_profile_small \
        ./laplace_mpi_cuda 256 256 1e-6 1000 sor 1
    
    # 大网格profiling
    mpirun -np 2 nsys profile -o laplace_profile_large \
        ./laplace_mpi_cuda 512 512 1e-6 1000 sor 1
    
    echo "生成报告: nsys stats laplace_profile_small.nsys-rep"

# 使用nvprof（CUDA 10及更早版本）
elif command -v nvprof &> /dev/null; then
    echo ""
    echo "=== nvprof profiling ==="
    
    # GPU kernel耗时分析
    mpirun -np 1 nvprof --print-gpu-trace \
        ./laplace_mpi_cuda 256 256 1e-6 1000 sor 1
    
    # GPU kernel分析
    mpirun -np 1 nvprof --analysis-metrics \
        ./laplace_mpi_cuda 256 256 1e-6 1000 sor 1
    
    # 保存到文件
    mpirun -np 1 nvprof -o laplace_profile.nvprof \
        ./laplace_mpi_cuda 256 256 1e-6 1000 sor 1
    
else
    echo "未找到nvprof或nsys，请安装CUDA Toolkit"
fi

echo ""
echo "=== MPI + CUDA 时间分析 ==="
mpirun -np 2 ./laplace_mpi_cuda 512 512 1e-6 5000 sor 1

echo ""
echo "=== 异步版本对比 ==="
echo -n "同步版本: "
mpirun -np 2 ./laplace_mpi_cuda 512 512 1e-6 5000 sor 1 0 2>/dev/null | grep "计算时间"

echo -n "异步版本: "
mpirun -np 2 ./laplace_mpi_cuda 512 512 1e-6 5000 sor 1 1 2>/dev/null | grep "计算时间"