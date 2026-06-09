/**
 * laplace_mpi_cuda - 二维Laplace方程MPI+CUDA并行求解程序
 * 
 * 功能：
 *   - 支持Jacobi、Gauss-Seidel、SOR三种迭代方法
 *   - MPI并行 + CUDA GPU加速
 *   - MPI并行IO输出（多进程写入同一文件）
 *   - 可选：Stream异步处理（IO与计算重叠）
 * 
 * 编译：详见CMakeLists.txt或Makefile
 * 运行：mpirun -np 2 ./laplace_mpi_cuda 256 256 1e-6 50000 sor
 * 
 * 参数说明：
 *   argv[1] - nx (x方向内点数)
 *   argv[2] - ny (y方向内点数)
 *   argv[3] - tol (收敛容差，默认1e-6)
 *   argv[4] - max_iters (最大迭代次数，默认50000)
 *   argv[5] - method (jacobi/gs/sor，默认sor)
 *   argv[6] - use_cuda (0/1，默认1)
 *   argv[7] - use_async (0/1，默认0，选做)
 * 
 * 边界条件：顶部为1，底/左/右为0，内部初始化为1（产生梯度）
 */

#include "laplace_solver.h"
#include <getopt.h>
#include <chrono>

void print_usage(const char *prog) {
    printf("用法: %s [选项]\n", prog);
    printf("选项:\n");
    printf("  -x <nx>           X方向内点数 (默认: 256)\n");
    printf("  -y <ny>           Y方向内点数 (默认: 256)\n");
    printf("  -t <tol>          收敛容差 (默认: 1e-6)\n");
    printf("  -i <max_iters>    最大迭代次数 (默认: 50000)\n");
    printf("  -m <method>       迭代方法: jacobi, gs, sor (默认: sor)\n");
    printf("  -c <0/1>          使用CUDA (默认: 1)\n");
    printf("  -a <0/1>          使用异步Stream (选做, 默认: 0)\n");
    printf("  -o <filename>     输出文件名 (默认: result.vtk)\n");
    printf("  -h                显示帮助\n");
}

int main(int argc, char *argv[]) {
    // ========== 1. 初始化MPI ==========
    int rank, size;
    init_mpi(&argc, &argv, &rank, &size);
    
    // ========== 2. 解析命令行参数 ==========
    int nx = 256, ny = 256;
    double tol = 1e-6;
    int max_iters = 50000;
    IterMethod method = SOR;
    int use_cuda = 1;
    int use_async = 0;
    char output_filename[256] = "result.vtk";
    
    // 简化版参数解析（支持位置参数）
    if (argc >= 3) {
        nx = atoi(argv[1]);
        ny = atoi(argv[2]);
    }
    if (argc >= 4) tol = atof(argv[3]);
    if (argc >= 5) max_iters = atoi(argv[4]);
    if (argc >= 6) {
        if (strcmp(argv[5], "jacobi") == 0) method = JACOBI;
        else if (strcmp(argv[5], "gs") == 0) method = GAUSS_SEIDEL;
        else if (strcmp(argv[5], "sor") == 0) method = SOR;
    }
    if (argc >= 7) use_cuda = atoi(argv[6]);
    if (argc >= 8) use_async = atoi(argv[7]);
    
    // 计算最优松弛因子（SOR）
    double pi = acos(-1.0);
    double omega = 2.0 / (1.0 + sin(pi / (nx + 1)));
    
    // ========== 3. 配置求解器 ==========
    SolverConfig cfg;
    init_config(&cfg, nx, ny, tol, max_iters, method, omega, rank, size);
    
    // ========== 4. 分配网格 ==========
    Grid grid;
    init_grid(&grid, cfg.nx_local, cfg.ny_local);
    
    // ========== 5. 初始化边界条件和初始场 ==========
    init_boundary(&grid, &cfg);
    set_initial_condition(&grid, &cfg);
    
    // ========== 6. 打印配置信息 ==========
    if (rank == 0) {
        printf("========================================\n");
        printf("二维Laplace方程MPI+CUDA并行求解\n");
        printf("网格规模: %d × %d (内点)\n", nx, ny);
        printf("MPI进程数: %d\n", size);
        printf("迭代方法: %s\n", 
               method == JACOBI ? "Jacobi" : 
               (method == GAUSS_SEIDEL ? "Gauss-Seidel" : "SOR"));
        if (method == SOR) printf("松弛因子 omega: %.6f\n", omega);
        printf("收敛容差: %.0e\n", tol);
        printf("最大迭代次数: %d\n", max_iters);
        printf("使用CUDA: %s\n", use_cuda ? "是" : "否");
        printf("异步Stream: %s\n", use_async ? "是" : "否");
        printf("========================================\n");
    }
    
    // ========== 7. 求解 ==========
    double elapsed;
    int final_iter = 0;
    double final_resid = 0.0;
    
    if (use_cuda) {
        // CUDA版本
        cudaStream_t stream = NULL;
        if (use_async) {
            cudaStreamCreate(&stream);
        }
        elapsed = solve_cuda(&grid, &cfg, stream);
        // 注意：solve_cuda需要返回最终迭代次数和残差
        // 为简化，这里假设solve_cuda内部打印结果
        if (stream) cudaStreamDestroy(stream);
    } else {
        // CPU版本
        elapsed = solve_cpu(&grid, &cfg);
    }
    
    // ========== 8. MPI并行IO输出 ==========
    if (use_async && use_cuda) {
        // 异步IO（选做）
        cudaStream_t io_stream;
        cudaStreamCreate(&io_stream);
        mpi_io_write_async(output_filename, &grid, &cfg, final_iter, io_stream);
        cudaStreamDestroy(io_stream);
    } else {
        mpi_io_write(output_filename, &grid, &cfg, final_iter);
    }
    
    // ========== 9. 清理 ==========
    free_grid(&grid);
    finalize_mpi();
    
    return 0;
}
