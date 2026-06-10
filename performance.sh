#!/bin/bash

# ============================================
# CloverLeaf 性能测试脚本
# 测试 MPI、OpenMP、Hybrid 三个版本
# ============================================

# 创建结果目录
RESULT_DIR="performance_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"

echo "=========================================="
echo "CloverLeaf 性能测试"
echo "结果保存目录: $RESULT_DIR"
echo "=========================================="

# ============================================
# MPI 版本测试
# ============================================
echo ""
echo "========== MPI 版本测试 =========="

cd CloverLeaf_MPI

for np in 1 2 4 8; do
    echo "运行 MPI $np 进程..."
    mpirun -np $np ./clover_leaf InputDecks/clover_bm_short.in 2>&1 | tee "../$RESULT_DIR/mpi_${np}p.txt"
done

cd ..

# ============================================
# OpenMP 版本测试
# ============================================
echo ""
echo "========== OpenMP 版本测试 =========="

cd CloverLeaf_OpenMP

for t in 1 2 4 8; do
    echo "运行 OpenMP $t 线程..."
    export OMP_NUM_THREADS=$t
    ./clover_leaf InputDecks/clover_bm_short.in 2>&1 | tee "../$RESULT_DIR/openmp_${t}t.txt"
done

cd ..

# ============================================
# Hybrid 版本测试
# ============================================
echo ""
echo "========== Hybrid 版本测试 =========="

cd CloverLeaf_ref

echo "运行 Hybrid 2进程 x 2线程..."
export OMP_NUM_THREADS=2
mpirun -np 2 ./clover_leaf InputDecks/clover_bm_short.in 2>&1 | tee "../$RESULT_DIR/hybrid_2p2t.txt"

echo "运行 Hybrid 4进程 x 2线程..."
export OMP_NUM_THREADS=2
mpirun -np 4 ./clover_leaf InputDecks/clover_bm_short.in 2>&1 | tee "../$RESULT_DIR/hybrid_4p2t.txt"

cd ..

# ============================================
# 生成汇总报告
# ============================================
echo ""
echo "=========================================="
echo "生成汇总报告..."
echo "=========================================="

cat > "$RESULT_DIR/summary.txt" << EOF2
CloverLeaf 性能测试汇总报告
测试时间: $(date)

=== MPI 测试结果 ===
EOF2

for np in 1 2 4 8; do
    time=$(grep "Wall clock" "$RESULT_DIR/mpi_${np}p.txt" 2>/dev/null | tail -1 | awk '{print $3}')
    echo "MPI $np 进程: $time 秒" >> "$RESULT_DIR/summary.txt"
done

echo "" >> "$RESULT_DIR/summary.txt"
echo "=== OpenMP 测试结果 ===" >> "$RESULT_DIR/summary.txt"

for t in 1 2 4 8; do
    time=$(grep "Wall clock" "$RESULT_DIR/openmp_${t}t.txt" 2>/dev/null | tail -1 | awk '{print $3}')
    echo "OpenMP $t 线程: $time 秒" >> "$RESULT_DIR/summary.txt"
done

echo "" >> "$RESULT_DIR/summary.txt"
echo "=== Hybrid 测试结果 ===" >> "$RESULT_DIR/summary.txt"

time1=$(grep "Wall clock" "$RESULT_DIR/hybrid_2p2t.txt" 2>/dev/null | tail -1 | awk '{print $3}')
time2=$(grep "Wall clock" "$RESULT_DIR/hybrid_4p2t.txt" 2>/dev/null | tail -1 | awk '{print $3}')
echo "Hybrid 2p2t: $time1 秒" >> "$RESULT_DIR/summary.txt"
echo "Hybrid 4p2t: $time2 秒" >> "$RESULT_DIR/summary.txt"

# 显示结果
echo ""
cat "$RESULT_DIR/summary.txt"

echo ""
echo "=========================================="
echo "测试完成！结果保存在: $RESULT_DIR/"
echo "=========================================="
