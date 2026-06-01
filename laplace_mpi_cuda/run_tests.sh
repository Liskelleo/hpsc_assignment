#!/bin/bash
# ============================================================
# 完整测试脚本 - Laplace MPI+CUDA求解器
# ============================================================

cd ~/laplace_mpi_cuda

echo "清理并编译..."
make clean
make

{
    echo "=========================================="
    echo "测试时间: $(date)"
    echo "=========================================="
    
    echo ""
    echo "=== 1. 正确性验证 ==="
    echo "对比不同迭代方法的结果"
    
    echo -n "Jacobi: "
    mpirun -np 2 ./laplace_mpi_cuda 128 128 1e-6 2000 jacobi 0 2>/dev/null | grep "计算时间"
    
    echo -n "Gauss-Seidel: "
    mpirun -np 2 ./laplace_mpi_cuda 128 128 1e-6 2000 gs 0 2>/dev/null | grep "计算时间"
    
    echo -n "SOR (omega=optimal): "
    mpirun -np 2 ./laplace_mpi_cuda 128 128 1e-6 2000 sor 0 2>/dev/null | grep "计算时间"
    
    echo ""
    echo "=== 2. CPU vs CUDA 性能对比 (512×512) ==="
    
    echo -n "CPU版本: "
    mpirun -np 2 ./laplace_mpi_cuda 512 512 1e-6 50000 sor 0 2>/dev/null | grep "计算时间"
    
    echo -n "CUDA版本: "
    mpirun -np 2 ./laplace_mpi_cuda 512 512 1e-6 50000 sor 1 2>/dev/null | grep "计算时间"
    
    echo ""
    echo "=== 3. Strong Scaling测试 (CUDA版本, 1024×1024) ==="
    for np in 1 2 4; do
        echo -n "进程数: $np | "
        mpirun -np $np ./laplace_mpi_cuda 1024 1024 1e-6 50000 sor 1 2>/dev/null | grep "计算时间"
    done
    
    echo ""
    echo "=== 4. Weak Scaling测试 (CUDA版本, 每进程256×256) ==="
    echo -n "进程数: 1 | 网格: 256×256 | "
    mpirun -np 1 ./laplace_mpi_cuda 256 256 1e-6 50000 sor 1 2>/dev/null | grep "计算时间"
    echo -n "进程数: 2 | 网格: 362×362 | "
    mpirun -np 2 ./laplace_mpi_cuda 362 362 1e-6 50000 sor 1 2>/dev/null | grep "计算时间"
    echo -n "进程数: 4 | 网格: 512×512 | "
    mpirun -np 4 ./laplace_mpi_cuda 512 512 1e-6 50000 sor 1 2>/dev/null | grep "计算时间"
    
    echo ""
    echo "=== 5. MPI并行IO测试 ==="
    mpirun -np 2 ./laplace_mpi_cuda 64 64 1e-6 500 sor 0 2>/dev/null | grep "计算时间"
    
    if [ -f "result.vtk" ]; then
        echo "VTK文件已生成: result.vtk"
        ls -lh result.vtk
    fi
    
    echo ""
    echo "=== 6. 异步Stream测试（选做） ==="
    echo -n "同步版本: "
    mpirun -np 2 ./laplace_mpi_cuda 512 512 1e-6 5000 sor 1 0 2>/dev/null | grep "计算时间"
    echo -n "异步版本: "
    mpirun -np 2 ./laplace_mpi_cuda 512 512 1e-6 5000 sor 1 1 2>/dev/null | grep "计算时间"
    
} | tee test_results.txt

echo ""
echo "测试完成！结果保存在 test_results.txt"