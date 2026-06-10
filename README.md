# 高性能科学计算期末大作业

本仓库为高性能科学计算课程期末大作业提交仓库，包含 CloverLeaf 程序阅读分析与二维 Laplace 方程 MPI+CUDA 并行求解两部分。

## 第一部分：CloverLeaf

目录：`CloverLeaf/`

该部分包含期末作业第一部分使用的 CloverLeaf 实际代码文件。作业要求围绕 CloverLeaf 的 Serial、MPI、OpenMP、MPI+OpenMP 以及 CUDA/OpenACC 版本开展编译、正确性验证、性能测试与并行效率分析。

## 第二部分：二维 Laplace 方程 MPI+CUDA 求解

目录：`laplace_mpi_cuda/`

该部分实现二维 Laplace 方程的 MPI+CUDA 并行求解程序，包含 Jacobi、Gauss-Seidel、SOR、MPI 并行 IO、CUDA kernel、Pinned Memory、异步 Stream 和单 GPU 全局 CUDA 版本等内容。

## 提交说明

源码、脚本、必要结果文件见各自分支。报告文件位于根目录，包括：

- `高性能计算期末大作业(刘汇川lpart1，3-7题)`
- `高性能科学计算大作业(刘尚坤part2)`
- `高性能科学计算大作业(杨译坤part1，1-4题)`

小组成员：

| 学号 | 姓名 |
| :--- | :--- |
| 2501111758 | 杨译坤 |
| 2501111747 | 刘汇川 |
| 2501111730 | 刘尚坤 |
