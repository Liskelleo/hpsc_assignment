#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p test_results
mkdir -p bin

{
    echo "=== CUDA Ghost Exchange: Pageable vs Pinned ==="
    echo "Date: $(date)"
    echo "Command baseline: CUDA SOR, tol=1e-6, max_iters=50000"
    echo
    for mode in pageable pinned; do
        for np in 1 2 4 8; do
            if [ "${mode}" = "pageable" ]; then
                echo -n "pageable np=${np} | "
                LAPLACE_USE_PAGEABLE_GHOST=1 mpirun -np "${np}" ./laplace_mpi_cuda 1024 1024 1e-6 50000 sor 1 0 "bin/pinned_pageable_1024x1024_np${np}.bin" 2>/dev/null | grep -E "完成|计算时间|输出文件" || true
            else
                echo -n "pinned   np=${np} | "
                mpirun -np "${np}" ./laplace_mpi_cuda 1024 1024 1e-6 50000 sor 1 0 "bin/pinned_pinned_1024x1024_np${np}.bin" 2>/dev/null | grep -E "完成|计算时间|输出文件" || true
            fi
        done
        echo
    done
} | tee test_results/pinned_memory_compare.txt
