# 高性能科学计算期末大作业

本仓库为高性能科学计算课程期末大作业提交仓库，包含 CloverLeaf 程序阅读分析与二维 Laplace 方程 MPI+CUDA 并行求解两部分。

## 第一部分：CloverLeaf

目录：`CloverLeaf/`

该部分包含期末作业第一部分使用的 CloverLeaf 实际代码文件。作业要求围绕 CloverLeaf 的 Serial、MPI、OpenMP、MPI+OpenMP 以及 CUDA/OpenACC 版本开展编译、正确性验证、性能测试与并行效率分析。

## 第二部分：二维 Laplace 方程 MPI+CUDA 求解

目录：`laplace_mpi_cuda/`

该部分实现二维 Laplace 方程的 MPI+CUDA 并行求解程序，包含 Jacobi、Gauss-Seidel、SOR、MPI 并行 IO、CUDA kernel、Pinned Memory、异步 Stream 和单 GPU 全局 CUDA 版本等内容。

主要文件：

- `laplace_mpi_cuda/doc/report.md`：第二部分正式报告
- `laplace_mpi_cuda/src/`：核心源代码
- `laplace_mpi_cuda/test_results/`：测试结果与可视化图片
- `laplace_mpi_cuda/bin/`：运行生成的二进制输出文件
- `laplace_mpi_cuda/run_tests.sh`、`scaling_compare.sh`、`size_scaling.sh`、`bench_pinned.sh`：测试脚本

## 提交说明

提交时请保留两个作业目录及其源码、脚本、报告和必要结果文件。最终压缩包按课程要求命名为“第XX小组期末大作业”。
