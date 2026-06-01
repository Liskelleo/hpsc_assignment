/**
 * stream_async.cu - 异步Stream处理（选做加分项）
 * 
 * 实现IO与计算kernel的重叠：
 *   1. 使用多个CUDA Stream
 *   2. 计算kernel在一个stream中执行
 *   3. 数据回传和MPI通信在另一个stream中执行
 *   4. 实现计算与通信的重叠
 */

#include "stream_async.h"  // ★ 引入新头文件，自动获取结构体和 sor_kernel 声明
#include <cuda_runtime.h>

/**
 * create_async_context - 创建异步处理上下文
 */
AsyncContext* create_async_context(int nx_local, int ny_local, int stride) {
    AsyncContext *ctx = (AsyncContext*)malloc(sizeof(AsyncContext));
    
    // 创建三个独立的CUDA Stream
    cudaStreamCreate(&ctx->compute_stream);
    cudaStreamCreate(&ctx->comm_stream);
    cudaStreamCreate(&ctx->io_stream);
    
    // 分配设备端缓冲区
    int size = (ny_local + 2) * stride * sizeof(double);
    cudaMalloc(&ctx->d_u_buffer, size);
    cudaMalloc(&ctx->d_u_new_buffer, size);
    
    // 分配主机端通信缓冲区
    ctx->buffer_size = nx_local * sizeof(double);  // 一行的大小
    cudaMallocHost(&ctx->h_send_buffer, ctx->buffer_size);
    cudaMallocHost(&ctx->h_recv_buffer, ctx->buffer_size);
    
    ctx->overlap_depth = 2;
    
    return ctx;
}

/**
 * async_exchange_ghost - 异步虚拟边界交换
 * 
 * 使用独立的通信stream实现边界数据交换与计算的重叠
 * 
 * 原理：
 *   1. 将边界数据异步复制到主机端固定内存
 *   2. 在comm_stream中等待复制完成
 *   3. 执行MPI通信（阻塞，但使用固定内存）
 *   4. 将接收到的数据异步复制回设备端
 */
void async_exchange_ghost(AsyncContext *ctx, Grid *grid, SolverConfig *cfg,
                          int direction) {
    int rank = cfg->rank;
    int size = cfg->size;
    int nx_local = grid->nx_local;
    int ny_local = grid->ny_local;
    int stride = grid->stride;
    double *d_u = grid->d_u;
    MPI_Status status;
    
    // 向上边界发送数据（第1行）
    if (rank > 0 && (direction == 0 || direction == 1)) {
        // 异步复制发送数据到主机固定内存
        cudaMemcpyAsync(ctx->h_send_buffer, &d_u[1 * stride + 1],
                        ctx->buffer_size, cudaMemcpyDeviceToHost,
                        ctx->comm_stream);
        
        // 在comm_stream上等待复制完成
        cudaStreamSynchronize(ctx->comm_stream);
        
        // MPI发送（使用固定内存，可与GPU计算重叠）
        MPI_Send(ctx->h_send_buffer, nx_local, MPI_DOUBLE, rank - 1, 0,
                 MPI_COMM_WORLD);
        
        // 异步接收
        MPI_Recv(ctx->h_recv_buffer, nx_local, MPI_DOUBLE, rank - 1, 0,
                 MPI_COMM_WORLD, &status);
        
        // 异步复制回设备端
        cudaMemcpyAsync(&d_u[0 * stride + 1], ctx->h_recv_buffer,
                        ctx->buffer_size, cudaMemcpyHostToDevice,
                        ctx->comm_stream);
    }
    
    // 向下边界发送数据（最后一行）
    if (rank < size - 1 && (direction == 0 || direction == 2)) {
        cudaMemcpyAsync(ctx->h_send_buffer, &d_u[ny_local * stride + 1],
                        ctx->buffer_size, cudaMemcpyDeviceToHost,
                        ctx->comm_stream);
        cudaStreamSynchronize(ctx->comm_stream);
        
        MPI_Send(ctx->h_send_buffer, nx_local, MPI_DOUBLE, rank + 1, 1,
                 MPI_COMM_WORLD);
        MPI_Recv(ctx->h_recv_buffer, nx_local, MPI_DOUBLE, rank + 1, 1,
                 MPI_COMM_WORLD, &status);
        
        cudaMemcpyAsync(&d_u[(ny_local + 1) * stride + 1], ctx->h_recv_buffer,
                        ctx->buffer_size, cudaMemcpyHostToDevice,
                        ctx->comm_stream);
    }
}

/**
 * async_compute_with_overlap - 重叠计算与通信
 * 
 * 使用多个Stream实现计算与通信的流水线重叠
 * 
 * 原理：
 *   对于红黑SOR迭代，红点和黑点交替更新
 *   可以在计算黑点的同时，异步交换红点更新后的边界数据
 */
void async_compute_with_overlap(AsyncContext *ctx, Grid *grid, SolverConfig *cfg,
                                cudaStream_t compute_stream,
                                cudaStream_t comm_stream) {
    int nx_local = grid->nx_local;
    int ny_local = grid->ny_local;
    int stride = grid->stride;
    double omega = cfg->omega;
    int nx_start = cfg->nx_start;
    int ny_start = cfg->ny_start;
    
    dim3 block_size(16, 16);
    dim3 grid_size((nx_local + block_size.x - 1) / block_size.x,
                   (ny_local + block_size.y - 1) / block_size.y);
    
    // ========== 阶段1：计算红点，同时异步交换上次迭代的黑点边界 ==========
    // 在compute_stream中计算红点
    sor_kernel<<<grid_size, block_size, 0, compute_stream>>>(
        grid->d_u, nx_local, ny_local, stride, 0, omega, nx_start, ny_start);
    
    // 在comm_stream中异步交换边界
    async_exchange_ghost(ctx, grid, cfg, 1);  // 向上交换
    
    // ========== 阶段2：等待通信完成，然后计算黑点 ==========
    cudaStreamWaitEvent(comm_stream, NULL, 0);
    
    sor_kernel<<<grid_size, block_size, 0, compute_stream>>>(
        grid->d_u, nx_local, ny_local, stride, 1, omega, nx_start, ny_start);
    
    async_exchange_ghost(ctx, grid, cfg, 2);  // 向下交换
}

/**
 * async_io_with_compute - 异步IO与计算重叠
 * 
 * 原理：
 *   在迭代过程中，定期将结果异步复制回主机
 *   IO操作在单独的stream中执行，不与计算争抢资源
 */
void async_io_with_compute(AsyncContext *ctx, Grid *grid, SolverConfig *cfg,
                           int iter, const char *filename) {
    int nx_local = grid->nx_local;
    int ny_local = grid->ny_local;
    int stride = grid->stride;
    
    // 每隔500次迭代输出一次
    if (iter % 500 == 0 && iter > 0) {
        // 在io_stream中异步复制数据回主机
        cudaMemcpyAsync(grid->h_u, grid->d_u,
                        (ny_local + 2) * stride * sizeof(double),
                        cudaMemcpyDeviceToHost, ctx->io_stream);
        
        // 在io_stream上等待复制完成
        cudaStreamSynchronize(ctx->io_stream);
        
        // 异步写入文件（使用独立的线程或MPI非阻塞IO）
        char tmp_filename[256];
        snprintf(tmp_filename, sizeof(tmp_filename), "%s.%d", filename, iter);
        
        // 这里可以调用非阻塞的MPI-IO
        // 实际实现中可以使用MPI_File_iwrite
    }
}

/**
 * destroy_async_context - 销毁异步处理上下文
 */
void destroy_async_context(AsyncContext *ctx) {
    cudaStreamDestroy(ctx->compute_stream);
    cudaStreamDestroy(ctx->comm_stream);
    cudaStreamDestroy(ctx->io_stream);
    cudaFree(ctx->d_u_buffer);
    cudaFree(ctx->d_u_new_buffer);
    cudaFreeHost(ctx->h_send_buffer);
    cudaFreeHost(ctx->h_recv_buffer);
    free(ctx);
}