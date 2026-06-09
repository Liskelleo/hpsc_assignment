#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p test_results
mkdir -p bin

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

{
    echo "=== Size Scaling: CPU vs MPI+CUDA vs Single-GPU CUDA ==="
    echo "Date: $(date)"
    echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
    echo
    for n in 256 512 1024 2048; do
        echo "=== grid=${n}x${n} ==="
        if [ "${n}" != "2048" ]; then
            echo -n "CPU np=1       | "
            mpirun -np 1 ./laplace_mpi_cuda "${n}" "${n}" 1e-6 50000 sor 0 0 "bin/size_${n}x${n}_np1_cpu.bin" 2>/dev/null | grep -E "SOR完成|计算时间|输出文件" || true
        else
            echo "CPU np=1       | skipped"
        fi

        echo -n "MPI+CUDA np=1  | "
        mpirun -np 1 ./laplace_mpi_cuda "${n}" "${n}" 1e-6 50000 sor 1 0 "bin/size_${n}x${n}_np1_cuda.bin" 2>/dev/null | grep -E "完成|计算时间|输出文件" || true

        echo -n "Single CUDA    | "
        ./laplace_cuda_single "${n}" "${n}" 1e-6 50000 "bin/size_${n}x${n}_single_cuda.bin" 2>/dev/null | grep -E "Single-GPU CUDA完成|计算时间|输出文件" || true
        echo
    done
} | tee test_results/size_scaling.txt
