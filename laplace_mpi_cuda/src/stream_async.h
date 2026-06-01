#ifndef STREAM_ASYNC_H
#define STREAM_ASYNC_H

#include "laplace_solver.h"
#include <cuda_runtime.h>

/**
 * AsyncContext - 异步处理上下文结构体声明
 */
typedef struct {
    cudaStream_t compute_stream;   // 计算流
    cudaStream_t comm_stream;      // 通信流
    cudaStream_t io_stream;        // IO流
    
    double *d_u_buffer;            // 设备端缓冲区
    double *d_u_new_buffer;
    double *h_send_buffer;         // 主机端发送缓冲区（MPI通信）
    double *h_recv_buffer;         // 主机端接收缓冲区
    
    int buffer_size;               // 缓冲区大小
    int overlap_depth;             // 重叠深度
} AsyncContext;

// 显式声明在 laplace_cuda.cu 中定义的 SOR 核函数，供异步流调用
extern __global__ void sor_kernel(double *u, int nx_local, int ny_local, int stride, 
                                  int color, double omega, int nx_start, int ny_start);

// 异步接口函数声明
AsyncContext* create_async_context(int nx_local, int ny_local, int stride);
void async_exchange_ghost(AsyncContext *ctx, Grid *grid, SolverConfig *cfg, int direction);
void async_compute_with_overlap(AsyncContext *ctx, Grid *grid, SolverConfig *cfg,
                                cudaStream_t compute_stream, cudaStream_t comm_stream);
void async_io_with_compute(AsyncContext *ctx, Grid *grid, SolverConfig *cfg,
                           int iter, const char *filename);
void destroy_async_context(AsyncContext *ctx);

#endif // STREAM_ASYNC_H