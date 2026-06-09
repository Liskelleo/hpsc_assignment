/**
 * laplace_cuda.cu - CUDA kernel实现
 * 
 * 包含Jacobi、Gauss-Seidel、SOR三种迭代方法的GPU实现
 * 红黑排序用于Gauss-Seidel和SOR的并行化
 */

#include "laplace_solver.h"

__device__ inline double atomicMaxDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(max(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

/* ============================================================
 * Jacobi迭代CUDA Kernel
 * 
 * 每个线程更新一个网格点，使用旧值计算新值
 * 网格划分：blockDim.x * gridDim.x = nx_local
 * ============================================================ */
__global__ void jacobi_kernel(double *u_new, const double *u, int nx_local, int ny_local, int stride) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;  // x方向索引（跳过左边界）
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;  // y方向索引（跳过上边界）
    
    if (i <= ny_local && j <= nx_local) {
        int idx = i * stride + j;
        // 五点差分格式
        u_new[idx] = 0.25 * (u[idx - 1] + u[idx + 1] + 
                             u[idx - stride] + u[idx + stride]);
    }
}

/* ============================================================
 * Gauss-Seidel迭代CUDA Kernel（红黑排序）
 * 
 * 红点（color=0）：i+j为偶数，使用黑点旧值
 * 黑点（color=1）：i+j为奇数，使用红点新值
 * 
 * 注意：Gauss-Seidel在GPU上需要两次kernel调用（红点和黑点分开）
 * ============================================================ */
__global__ void gs_kernel(double *u, int nx_local, int ny_local, int stride, int color, int nx_start, int ny_start) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i <= ny_local && j <= nx_local) {
        int global_i = i + ny_start - 1;
        int global_j = j + nx_start - 1;
        
        // 只处理指定颜色的点
        if ((global_i + global_j) % 2 == color) {
            int idx = i * stride + j;
            // 使用当前已更新的邻居值（红黑排序保证正确性）
            u[idx] = 0.25 * (u[idx - 1] + u[idx + 1] + 
                             u[idx - stride] + u[idx + stride]);
        }
    }
}

/* ============================================================
 * SOR迭代CUDA Kernel（红黑排序 + 松弛因子）
 * 
 * 在Gauss-Seidel基础上加入松弛因子omega加速收敛
 * ============================================================ */
__global__ void sor_kernel(double *u, int nx_local, int ny_local, int stride, 
                           int color, double omega, int nx_start, int ny_start) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i <= ny_local && j <= nx_local) {
        int global_i = i + ny_start - 1;
        int global_j = j + nx_start - 1;
        
        if ((global_i + global_j) % 2 == color) {
            int idx = i * stride + j;
            double old_val = u[idx];
            double relaxation = 0.25 * (u[idx - 1] + u[idx + 1] + 
                                        u[idx - stride] + u[idx + stride]);
            u[idx] = (1.0 - omega) * old_val + omega * relaxation;
        }
    }
}

/* ============================================================
 * 残差计算CUDA Kernel
 * 
 * 计算当前解与精确解的误差，使用atomicMax规约
 * ============================================================ */
__global__ void residual_kernel(double *u, double *residual, int nx_local, int ny_local, int stride) {
    __shared__ double shared_resid[256];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    double local_resid = 0.0;
    
    if (i <= ny_local && j <= nx_local) {
        int idx = i * stride + j;
        double relaxation = 0.25 * (u[idx - 1] + u[idx + 1] + 
                                    u[idx - stride] + u[idx + stride]);
        double diff = fabs(relaxation - u[idx]);
        if (!isfinite(diff)) diff = INFINITY;
        if (diff > local_resid) local_resid = diff;
    }
    
    // 共享内存规约
    shared_resid[tid] = local_resid;
    __syncthreads();
    
    int block_threads = blockDim.x * blockDim.y;
    for (int s = block_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_resid[tid] < shared_resid[tid + s])
                shared_resid[tid] = shared_resid[tid + s];
        }
        __syncthreads();
    }
    
    // 原子操作更新全局残差
    if (tid == 0) {
        // atomicMax(residual, (int)(shared_resid[0] * 1e10));
        // 直接进行真实的 double 类型的原子最大值更新，不需要再乘 1e10 和强转 int 了！
        atomicMaxDouble(residual, shared_resid[0]);
    }
}

/* ============================================================
 * GPU内存复制与通信辅助函数
 * ============================================================ */

/**
 * copy_to_device - 将主机数据复制到设备
 */
void copy_to_device(Grid *grid) {
    int size = (grid->ny_local + 2) * grid->stride * sizeof(double);
    cudaMemcpy(grid->d_u, grid->h_u, size, cudaMemcpyHostToDevice);
    cudaMemcpy(grid->d_u_new, grid->h_u_new, size, cudaMemcpyHostToDevice);
}

/**
 * copy_to_host - 将设备数据复制到主机
 */
void copy_to_host(Grid *grid) {
    int size = (grid->ny_local + 2) * grid->stride * sizeof(double);
    cudaMemcpy(grid->h_u, grid->d_u, size, cudaMemcpyDeviceToHost);
}

/**
 * exchange_ghost_cuda - GPU上的虚拟边界交换
 * 
 * 注意：CUDA kernel不能直接调用MPI，需要先复制到主机
 */
void exchange_ghost_cuda(Grid *grid, SolverConfig *cfg, cudaStream_t stream) {
    MPI_Status status;
    int rank = cfg->rank;
    int size = cfg->size;
    int nx_local = grid->nx_local;
    int ny_local = grid->ny_local;
    int stride = grid->stride;
    size_t row_bytes = (size_t)nx_local * sizeof(double);

    int use_pageable = getenv("LAPLACE_USE_PAGEABLE_GHOST") != NULL;
    double *send_buf = NULL;
    double *recv_buf = NULL;

    if (use_pageable) {
        send_buf = (double *)malloc(row_bytes);
        recv_buf = (double *)malloc(row_bytes);
    } else {
        if (!grid->h_pin_row_snd) cudaMallocHost(&grid->h_pin_row_snd, row_bytes);
        if (!grid->h_pin_row_rcv) cudaMallocHost(&grid->h_pin_row_rcv, row_bytes);
        send_buf = grid->h_pin_row_snd;
        recv_buf = grid->h_pin_row_rcv;
    }

    // 向上发送第1行，接收作为第0行。
    if (rank > 0) {
        cudaMemcpyAsync(send_buf, &grid->d_u[1 * stride + 1],
                        row_bytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        MPI_Sendrecv(send_buf, nx_local, MPI_DOUBLE, rank - 1, 0,
                     recv_buf, nx_local, MPI_DOUBLE, rank - 1, 0,
                     MPI_COMM_WORLD, &status);
        cudaMemcpyAsync(&grid->d_u[0 * stride + 1], recv_buf,
                        row_bytes, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    // 向下发送最后一行，接收作为第ny_local+1行。
    if (rank < size - 1) {
        cudaMemcpyAsync(send_buf, &grid->d_u[ny_local * stride + 1],
                        row_bytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        MPI_Sendrecv(send_buf, nx_local, MPI_DOUBLE, rank + 1, 0,
                     recv_buf, nx_local, MPI_DOUBLE, rank + 1, 0,
                     MPI_COMM_WORLD, &status);
        cudaMemcpyAsync(&grid->d_u[(ny_local + 1) * stride + 1], recv_buf,
                        row_bytes, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    if (use_pageable) {
        free(send_buf);
        free(recv_buf);
    }
}

/* ============================================================
 * CUDA求解器主函数
 * ============================================================ */

double solve_cuda(Grid *grid, SolverConfig *cfg, cudaStream_t stream) {
    // 分配设备内存
    int size = (cfg->ny_local + 2) * (cfg->nx_local + 2) * sizeof(double);
    cudaMalloc(&grid->d_u, size);
    cudaMalloc(&grid->d_u_new, size);
    
    // 复制初始条件到设备
    copy_to_device(grid);
    
    // 配置CUDA kernel的网格和块大小
    dim3 block_size(16, 16);
    dim3 grid_size((cfg->nx_local + block_size.x - 1) / block_size.x,
                   (cfg->ny_local + block_size.y - 1) / block_size.y);
    
    double *d_residual;
    cudaMalloc(&d_residual, sizeof(double));
    
    double start_time = MPI_Wtime();
    int iter = 0;
    double global_resid = cfg->tol + 1.0;
    
    while (global_resid > cfg->tol && iter < cfg->max_iters) {
        
        if (cfg->method == JACOBI) {
            // Jacobi迭代
            jacobi_kernel<<<grid_size, block_size, 0, stream>>>(
                grid->d_u_new, grid->d_u, cfg->nx_local, cfg->ny_local, grid->stride);
            
            // 交换指针
            double *temp = grid->d_u;
            grid->d_u = grid->d_u_new;
            grid->d_u_new = temp;

            exchange_ghost_cuda(grid, cfg, stream);
            
        } else if (cfg->method == GAUSS_SEIDEL) {
            // Gauss-Seidel迭代（红黑排序）
            for (int color = 0; color < 2; color++) {
                gs_kernel<<<grid_size, block_size, 0, stream>>>(
                    grid->d_u, cfg->nx_local, cfg->ny_local, grid->stride, 
                    color, cfg->nx_start, cfg->ny_start);
                exchange_ghost_cuda(grid, cfg, stream);
            }
            
        } else if (cfg->method == SOR) {
            // SOR迭代（红黑排序 + 松弛因子）
            for (int color = 0; color < 2; color++) {
                sor_kernel<<<grid_size, block_size, 0, stream>>>(
                    grid->d_u, cfg->nx_local, cfg->ny_local, grid->stride,
                    color, cfg->omega, cfg->nx_start, cfg->ny_start);
                exchange_ghost_cuda(grid, cfg, stream);
            }
        }
        
        iter++;
        
        // 每50次迭代计算残差
        if (iter % 50 == 0) {
            cudaMemset(d_residual, 0, sizeof(double));
            residual_kernel<<<grid_size, block_size, 0, stream>>>(
                grid->d_u, d_residual, cfg->nx_local, cfg->ny_local, grid->stride);
            
            double local_resid;
            cudaMemcpyAsync(&local_resid, d_residual, sizeof(double), 
                           cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            
            MPI_Allreduce(&local_resid, &global_resid, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            
            if (cfg->rank == 0 && iter % 500 == 0) {
                printf("迭代 %d, 残差 = %.2e\n", iter, global_resid);
            }
        }
    }
    
    double end_time = MPI_Wtime();
    
    // 复制最终结果回主机
    copy_to_host(grid);
    
    if (cfg->rank == 0) {
        printf("完成! 计算时间: %.4f 秒 (迭代: %d, 残差: %.2e)\n", 
               end_time - start_time, iter, global_resid);
    }
    
    // 清理设备内存
    cudaFree(grid->d_u);
    cudaFree(grid->d_u_new);
    cudaFree(d_residual);
    
    return end_time - start_time;
}
