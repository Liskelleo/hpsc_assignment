#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p test_results

echo "清理并编译..."
make clean
make

{
    echo "=== Full Test Results ==="
    echo "Date: $(date)"
    echo

    echo "=== 1. Correctness / method comparison, 128x128 ==="
    echo -n "Jacobi       | "
    mpirun -np 2 ./laplace_mpi_cuda 128 128 1e-6 2000 jacobi 0 2>/dev/null | grep -E "Jacobi完成|计算时间" || true
    echo -n "Gauss-Seidel | "
    mpirun -np 2 ./laplace_mpi_cuda 128 128 1e-6 2000 gs 0 2>/dev/null | grep -E "Gauss-Seidel完成|计算时间" || true
    echo -n "SOR          | "
    mpirun -np 2 ./laplace_mpi_cuda 128 128 1e-6 2000 sor 0 2>/dev/null | grep -E "SOR完成|计算时间" || true
    echo

    echo "=== 2. CPU vs CUDA, 512x512 ==="
    echo -n "CPU SOR       | "
    mpirun -np 2 ./laplace_mpi_cuda 512 512 1e-6 50000 sor 0 2>/dev/null | grep -E "SOR完成|计算时间" || true
    echo -n "CUDA SOR sync | "
    mpirun -np 2 ./laplace_mpi_cuda 512 512 1e-6 50000 sor 1 0 2>/dev/null | grep -E "完成|计算时间" || true
    echo -n "CUDA async    | "
    mpirun -np 2 ./laplace_mpi_cuda 512 512 1e-6 50000 sor 1 1 2>/dev/null | grep -E "完成|计算时间" || true
    echo

    echo "=== 3. CPU vs CUDA, 1024x1024 np=1 ==="
    echo -n "CPU SOR np=1       | "
    mpirun -np 1 ./laplace_mpi_cuda 1024 1024 1e-6 50000 sor 0 2>/dev/null | grep -E "SOR完成|计算时间" || true
    echo -n "CUDA SOR np=1      | "
    mpirun -np 1 ./laplace_mpi_cuda 1024 1024 1e-6 50000 sor 1 2>/dev/null | grep -E "完成|计算时间" || true
    echo -n "Single-GPU CUDA    | "
    ./laplace_cuda_single 1024 1024 1e-6 50000 test_results/single_cuda_1024x1024.bin 2>/dev/null | grep -E "Single-GPU CUDA完成|计算时间" || true
    echo

    echo "=== 4. MPI-IO validation, 512x512 ==="
    mpirun -np 2 ./laplace_mpi_cuda 512 512 1e-6 50000 sor 0 2>/dev/null | grep -E "SOR完成|计算时间" || true
    mv -f result.vtk test_results/result_sor_512x512_i50000.bin
    ./check_mpi_io.py test_results/result_sor_512x512_i50000.bin --nx 512 --ny 512
    echo

    echo "=== 5. Figure generation ==="
    ./visualize.py test_results
    echo "test_results/convergence_frames.png"
    echo "test_results/multi_resolution_comparison.png"
    echo "test_results/result_sor_1024x1024_i50000.png"
    echo "test_results/result_single_cuda_512x512_i50000.png"
} | tee test_results/full_test_results.txt

echo
echo "测试完成！结果保存在 test_results/full_test_results.txt"
