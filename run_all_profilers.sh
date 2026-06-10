#!/bin/bash

echo "=========================================="
echo "CloverLeaf 四个版本 Profiler 性能分析"
echo "=========================================="
echo ""

# 创建结果目录
mkdir -p profiler_results

# 1. Serial 版本
echo ">>> 1/4 正在测试 Serial 版本..."
cd CloverLeaf_Serial
/usr/bin/time -v ./clover_leaf InputDecks/clover_bm_short.in 2>&1 | tee ../profiler_results/serial_profiler.txt
cd ..

# 2. MPI 版本（4进程）
echo ""
echo ">>> 2/4 正在测试 MPI 版本（4进程）..."
cd CloverLeaf_MPI
/usr/bin/time -v mpirun -np 4 ./clover_leaf InputDecks/clover_bm_short.in 2>&1 | tee ../profiler_results/mpi_profiler.txt
cd ..

# 3. OpenMP 版本（4线程）
echo ""
echo ">>> 3/4 正在测试 OpenMP 版本（4线程）..."
cd CloverLeaf_OpenMP
export OMP_NUM_THREADS=4
/usr/bin/time -v ./clover_leaf InputDecks/clover_bm_short.in 2>&1 | tee ../profiler_results/openmp_profiler.txt
cd ..

# 4. Hybrid 版本（2进程×2线程）
echo ""
echo ">>> 4/4 正在测试 Hybrid 版本（2进程×2线程）..."
cd CloverLeaf_ref
export OMP_NUM_THREADS=2
/usr/bin/time -v mpirun -np 2 ./clover_leaf InputDecks/clover_bm_short.in 2>&1 | tee ../profiler_results/hybrid_profiler.txt
cd ..

echo ""
echo "=========================================="
echo "所有 Profiler 测试完成！"
echo "结果保存在 profiler_results/ 目录"
echo "=========================================="

# 汇总结果
echo ""
echo "========== 性能指标汇总 =========="
echo ""

for version in serial mpi openmp hybrid; do
    echo "--- ${version} ---"
    grep -E "User time|System time|Percent of CPU|Maximum resident" profiler_results/${version}_profiler.txt 2>/dev/null | head -4
    echo ""
done
