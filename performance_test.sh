#!/bin/bash

# ============================================
# CloverLeaf 性能测试脚本
# 测试 MPI、OpenMP、Hybrid 三个版本
# 小算例: 960×960, 87步
# 大算例: 3840×3840, 87步
# ============================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 结果目录
RESULT_DIR="performance_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"

echo -e "${GREEN}=========================================="
echo "CloverLeaf 性能测试"
echo "测试开始时间: $(date)"
echo "结果保存目录: $RESULT_DIR"
echo -e "==========================================${NC}"

# ============================================
# 函数：运行测试并保存结果
# ============================================
run_test() {
    local name=$1
    local cmd=$2
    local output_file="$RESULT_DIR/${name}.txt"
    
    echo -e "${YELLOW}正在运行: $name${NC}"
    echo "命令: $cmd"
    
    eval "$cmd" > "$output_file" 2>&1
    
    # 提取运行时间
    local wall_time=$(grep "Wall clock" "$output_file" | tail -1 | awk '{print $3}')
    echo -e "${GREEN}完成: $name, 时间: $wall_time 秒${NC}"
    echo ""
}

# ============================================
# 1. MPI 版本测试
# ============================================
echo -e "${BLUE}========== MPI 版本测试 ==========${NC}"

# MPI 小算例
cd CloverLeaf_MPI
for np in 1 2 4 8; do
    run_test "mpi_small_${np}p" "mpirun -np $np ./clover_leaf InputDecks/clover_bm_short.in"
done

# MPI 大算例
for np in 1 2 4 8; do
    run_test "mpi_large_${np}p" "mpirun -np $np ./clover_leaf InputDecks/clover_bm16_short.in"
done
cd ..

# ============================================
# 2. OpenMP 版本测试
# ============================================
echo -e "${BLUE}========== OpenMP 版本测试 ==========${NC}"

cd CloverLeaf_OpenMP
for t in 1 2 4 8; do
    export OMP_NUM_THREADS=$t
    run_test "openmp_small_${t}t" "./clover_leaf InputDecks/clover_bm_short.in"
    run_test "openmp_large_${t}t" "./clover_leaf InputDecks/clover_bm16_short.in"
done
cd ..

# ============================================
# 3. Hybrid (MPI+OpenMP) 版本测试
# ============================================
echo -e "${BLUE}========== Hybrid 版本测试 ==========${NC}"

cd CloverLeaf_ref

# Hybrid 小算例
export OMP_NUM_THREADS=2
run_test "hybrid_small_1p2t" "mpirun -np 1 ./clover_leaf InputDecks/clover_bm_short.in"

export OMP_NUM_THREADS=4
run_test "hybrid_small_1p4t" "mpirun -np 1 ./clover_leaf InputDecks/clover_bm_short.in"

export OMP_NUM_THREADS=1
run_test "hybrid_small_2p1t" "mpirun -np 2 ./clover_leaf InputDecks/clover_bm_short.in"

export OMP_NUM_THREADS=2
run_test "hybrid_small_2p2t" "mpirun -np 2 ./clover_leaf InputDecks/clover_bm_short.in"

export OMP_NUM_THREADS=4
run_test "hybrid_small_2p4t" "mpirun -np 2 ./clover_leaf InputDecks/clover_bm_short.in"

export OMP_NUM_THREADS=1
run_test "hybrid_small_4p1t" "mpirun -np 4 ./clover_leaf InputDecks/clover_bm_short.in"

export OMP_NUM_THREADS=2
run_test "hybrid_small_4p2t" "mpirun -np 4 ./clover_leaf InputDecks/clover_bm_short.in"

# Hybrid 大算例
export OMP_NUM_THREADS=2
run_test "hybrid_large_1p2t" "mpirun -np 1 ./clover_leaf InputDecks/clover_bm16_short.in"

export OMP_NUM_THREADS=4
run_test "hybrid_large_1p4t" "mpirun -np 1 ./clover_leaf InputDecks/clover_bm16_short.in"

export OMP_NUM_THREADS=1
run_test "hybrid_large_2p1t" "mpirun -np 2 ./clover_leaf InputDecks/clover_bm16_short.in"

export OMP_NUM_THREADS=2
run_test "hybrid_large_2p2t" "mpirun -np 2 ./clover_leaf InputDecks/clover_bm16_short.in"

export OMP_NUM_THREADS=4
run_test "hybrid_large_2p4t" "mpirun -np 2 ./clover_leaf InputDecks/clover_bm16_short.in"

export OMP_NUM_THREADS=1
run_test "hybrid_large_4p1t" "mpirun -np 4 ./clover_leaf InputDecks/clover_bm16_short.in"

export OMP_NUM_THREADS=2
run_test "hybrid_large_4p2t" "mpirun -np 4 ./clover_leaf InputDecks/clover_bm16_short.in"

cd ..

# ============================================
# 生成汇总报告
# ============================================
echo -e "${GREEN}=========================================="
echo "测试完成！生成汇总报告..."
echo -e "==========================================${NC}"

cat > "$RESULT_DIR/summary.txt" << 'EOF'
# CloverLeaf 性能测试汇总报告

## 测试配置
- 小算例: clover_bm_short.in (960×960, 87步)
- 大算例: clover_bm16_short.in (3840×3840, 87步)
- 测试时间: $(date)

## MPI 测试结果（小算例）
