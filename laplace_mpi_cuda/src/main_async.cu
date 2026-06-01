/**
 * main_async.cu - 异步Stream版本主程序
 * 
 * 实现计算与IO的完全重叠
 * 编译：make test_async
 */

#include "laplace_solver.h"
#include "stream_async.h"  // ★ 必须加上这一行，否则 main 无法识别 AsyncContext

int main(int argc, char *argv[]) {
    // ========== 1. 初始化MPI ==========
    int rank, size;
    init_mpi(&argc, &argv, &rank, &size);
    
    // ========== 2. 解析参数 ==========
    int nx = (argc >= 3) ? atoi(argv[1]) : 256;
    int ny = (argc >= 3) ? atoi(argv[2]) : 256;
    double tol = (argc >= 4) ? atof(argv[3]) : 1e-6;
    int max_iters = (argc >= 5) ? atoi(argv[4]) : 50000;
    IterMethod method = SOR;
    
    // ========== 3. 配置求解器 ==========
    SolverConfig cfg;
    double pi = acos(-1.0);
    double omega = 2.0 / (1.0 + sin(pi / (nx + 1)));
    init_config(&cfg, nx, ny, tol, max_iters, method, omega, rank, size);
    
    // ========== 4. 分配网格 ==========
    Grid grid;
    init_grid(&grid, cfg.nx_local, cfg.ny_local);
    init_boundary(&grid, &cfg);
    set_initial_condition(&grid, &cfg);
    
    // ========== 5. 创建异步上下文 ==========
    AsyncContext *ctx = create_async_context(cfg.nx_local, cfg.ny_local, grid.stride);
    
    // 复制初始数据到设备
    cudaMemcpyAsync(ctx->d_u_buffer, grid.h_u,
                    (cfg.ny_local + 2) * grid.stride * sizeof(double),
                    cudaMemcpyHostToDevice, ctx->compute_stream);
    
    double start_time = MPI_Wtime();
    int iter = 0;
    double resid = tol + 1.0;
    
    // ========== 6. 异步迭代求解 ==========
    while (resid > tol && iter < max_iters) {
        // 使用重叠计算与通信
        async_compute_with_overlap(ctx, &grid, &cfg,
                                   ctx->compute_stream, ctx->comm_stream);
        
        iter++;
        
        // 定期计算残差并异步输出
        if (iter % 50 == 0) {
            // 在计算stream中同步，确保数据完整
            cudaStreamSynchronize(ctx->compute_stream);
            resid = compute_residual_cpu(&grid, &cfg);
            
            // 异步IO（不阻塞计算）
            async_io_with_compute(ctx, &grid, &cfg, iter, "result_async.vtk");
            
            if (rank == 0 && iter % 500 == 0) {
                printf("迭代 %d, 残差 = %.2e\n", iter, resid);
            }
        }
    }
    
    double end_time = MPI_Wtime();
    
    // 等待所有stream完成
    cudaStreamSynchronize(ctx->compute_stream);
    cudaStreamSynchronize(ctx->comm_stream);
    cudaStreamSynchronize(ctx->io_stream);
    
    // 最终输出
    mpi_io_write("result_async_final.vtk", &grid, &cfg, iter);
    
    if (rank == 0) {
        printf("异步版本完成! 计算时间: %.4f 秒 (迭代: %d, 残差: %.2e)\n",
               end_time - start_time, iter, resid);
    }
    
    // 清理
    destroy_async_context(ctx);
    free_grid(&grid);
    finalize_mpi();
    
    return 0;
}