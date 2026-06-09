#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p test_results

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

{
    echo "=== Scaling Compare: CPU vs CUDA ==="
    echo "Date: $(date)"
    echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
    echo
    echo "=== Strong Scaling: fixed 1024x1024, np=1,2,4,8 ==="
    for np in 1 2 4 8; do
        echo -n "CPU  np=${np} | "
        mpirun -np "${np}" ./laplace_mpi_cuda 1024 1024 1e-6 50000 sor 0 2>/dev/null | grep -E "SOR完成|完成|计算时间" || true
        echo -n "CUDA np=${np} | "
        mpirun -np "${np}" ./laplace_mpi_cuda 1024 1024 1e-6 50000 sor 1 2>/dev/null | grep -E "SOR完成|完成|计算时间" || true
    done
    echo
    echo "=== Weak Scaling: about 256x256 per process, np=1,2,4,8 ==="
    for item in "1 256 256" "2 362 362" "4 512 512" "8 724 724"; do
        set -- ${item}
        np=$1; nx=$2; ny=$3
        echo -n "CPU  np=${np} grid=${nx}x${ny} | "
        mpirun -np "${np}" ./laplace_mpi_cuda "${nx}" "${ny}" 1e-6 50000 sor 0 2>/dev/null | grep -E "SOR完成|完成|计算时间" || true
        echo -n "CUDA np=${np} grid=${nx}x${ny} | "
        mpirun -np "${np}" ./laplace_mpi_cuda "${nx}" "${ny}" 1e-6 50000 sor 1 2>/dev/null | grep -E "SOR完成|完成|计算时间" || true
    done
} | tee test_results/scaling_compare_cpu_cuda.txt
