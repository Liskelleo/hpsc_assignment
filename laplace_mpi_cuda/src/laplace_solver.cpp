/**
 * laplace_solver.cpp - CPU求解器实现
 * 
 * 包含Jacobi、Gauss-Seidel、SOR三种迭代方法的CPU版本
 * 红黑排序用于Gauss-Seidel和SOR的并行化（多线程使用OpenMP）
 */

#include "laplace_solver.h"
#include <omp.h>
#include <cmath>

/* ============================================================
 * MPI初始化与区域分解
 * ============================================================ */

void init_mpi(int *argc, char ***argv, int *rank, int *size) {
    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
    MPI_Comm_size(MPI_COMM_WORLD, size);
}

/**
 * get_decomp_1d - 一维区域分解负载分配
 * 
 * 将ny行网格均匀分配给size个进程
 * 多余的行（ny % size）依次分配给前几个进程
 */
void get_decomp_1d(int ny, int rank, int size, int *ny_local, int *ny_start) {
    int base = ny / size;
    int rem = ny % size;
    
    if (rank < rem) {
        *ny_local = base + 1;
        *ny_start = rank * (base + 1) + 1;
    } else {
        *ny_local = base;
        *ny_start = rank * base + rem + 1;
    }
}

void init_config(SolverConfig *cfg, int nx, int ny, double tol, int max_iters,
                 IterMethod method, double omega, int rank, int size) {
    cfg->nx = nx;
    cfg->ny = ny;
    cfg->tol = tol;
    cfg->max_iters = max_iters;
    cfg->method = method;
    cfg->omega = omega;
    cfg->rank = rank;
    cfg->size = size;
    
    cfg->dx = 1.0 / (nx + 1);
    cfg->dy = 1.0 / (ny + 1);
    
    // 一维区域分解（沿Y方向）
    get_decomp_1d(ny, rank, size, &cfg->ny_local, &cfg->ny_start);
    cfg->nx_local = nx;
    cfg->nx_start = 1;
}

void init_grid(Grid *grid, int nx_local, int ny_local) {
    grid->nx_local = nx_local;
    grid->ny_local = ny_local;
    grid->stride = nx_local + 2;
    
    int size = (ny_local + 2) * grid->stride;
    grid->h_u = (double *)calloc(size, sizeof(double));
    grid->h_u_new = (double *)calloc(size, sizeof(double));
    grid->d_u = NULL;
    grid->d_u_new = NULL;
}

/**
 * init_boundary - 初始化狄利克雷边界条件
 * 
 * 四周边界均为0
 */
void init_boundary(Grid *grid, SolverConfig *cfg) {
    int nx_local = grid->nx_local;
    int ny_local = grid->ny_local;
    int stride = grid->stride;
    double *u = grid->h_u;
    double *u_new = grid->h_u_new;
    int rank = cfg->rank;
    int size = cfg->size;
    
    // 上边界（全局第一行，只有rank=0的进程有）
    if (rank == 0) {
        for (int j = 0; j <= nx_local + 1; j++) {
            u[0 * stride + j] = 0.0;
            u_new[0 * stride + j] = 0.0;
        }
    }
    
    // 下边界（全局最后一行，只有rank=size-1的进程有）
    if (rank == size - 1) {
        for (int j = 0; j <= nx_local + 1; j++) {
            u[(ny_local + 1) * stride + j] = 0.0;
            u_new[(ny_local + 1) * stride + j] = 0.0;
        }
    }
    
    // 左右边界（所有进程）
    for (int i = 0; i <= ny_local + 1; i++) {
        u[i * stride + 0] = 0.0;
        u[i * stride + nx_local + 1] = 0.0;
        u_new[i * stride + 0] = 0.0;
        u_new[i * stride + nx_local + 1] = 0.0;
    }
}

/**
 * set_initial_condition - 设置初始条件
 * 
 * 内部点初始化为1.0（产生梯度，驱动迭代）
 */
void set_initial_condition(Grid *grid, SolverConfig *cfg) {
    int nx_local = grid->nx_local;
    int ny_local = grid->ny_local;
    int stride = grid->stride;
    double *u = grid->h_u;
    
    for (int i = 1; i <= ny_local; i++) {
        for (int j = 1; j <= nx_local; j++) {
            u[i * stride + j] = 1.0;
        }
    }
}

/* ============================================================
 * 虚拟边界交换（MPI通信）
 * ============================================================ */

/**
 * exchange_ghost_mpi - 交换虚拟边界（一维分解）
 * 
 * 向上边界进程发送第1行，同时接收作为自己的第0行
 * 向下边界进程发送最后一行，同时接收作为自己的第ny_local+1行
 */
void exchange_ghost_mpi(Grid *grid, SolverConfig *cfg) {
    MPI_Status status;
    int rank = cfg->rank;
    int size = cfg->size;
    int nx_local = grid->nx_local;
    int ny_local = grid->ny_local;
    int stride = grid->stride;
    double *u = grid->h_u;
    
    // 向上发送/接收
    if (rank > 0) {
        MPI_Sendrecv(&u[1 * stride + 1], nx_local, MPI_DOUBLE, rank - 1, 0,
                     &u[0 * stride + 1], nx_local, MPI_DOUBLE, rank - 1, 0,
                     MPI_COMM_WORLD, &status);
    }
    
    // 向下发送/接收
    if (rank < size - 1) {
        MPI_Sendrecv(&u[ny_local * stride + 1], nx_local, MPI_DOUBLE, rank + 1, 0,
                     &u[(ny_local + 1) * stride + 1], nx_local, MPI_DOUBLE, rank + 1, 0,
                     MPI_COMM_WORLD, &status);
    }
}

/* ============================================================
 * 残差计算
 * ============================================================ */

double compute_residual_cpu(Grid *grid, SolverConfig *cfg) {
    double local_resid = 0.0, global_resid;
    int nx_local = grid->nx_local;
    int ny_local = grid->ny_local;
    int stride = grid->stride;
    double *u = grid->h_u;
    
    for (int i = 1; i <= ny_local; i++) {
        for (int j = 1; j <= nx_local; j++) {
            int idx = i * stride + j;
            double relaxation = 0.25 * (u[idx - 1] + u[idx + 1] +
                                        u[idx - stride] + u[idx + stride]);
            double diff = fabs(relaxation - u[idx]);
            if (diff > local_resid) local_resid = diff;
        }
    }
    
    MPI_Allreduce(&local_resid, &global_resid, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return global_resid;
}

/* ============================================================
 * Jacobi迭代（CPU版本）
 * ============================================================ */

double solve_jacobi_cpu(Grid *grid, SolverConfig *cfg) {
    int nx_local = grid->nx_local;
    int ny_local = grid->ny_local;
    int stride = grid->stride;
    double *u = grid->h_u;
    double *u_new = grid->h_u_new;
    int rank = cfg->rank;
    int iter = 0;
    double resid = cfg->tol + 1.0;
    
    double start_time = MPI_Wtime();
    
    while (resid > cfg->tol && iter < cfg->max_iters) {
        // Jacobi迭代：使用旧值计算新值
        for (int i = 1; i <= ny_local; i++) {
            for (int j = 1; j <= nx_local; j++) {
                int idx = i * stride + j;
                u_new[idx] = 0.25 * (u[idx - 1] + u[idx + 1] +
                                     u[idx - stride] + u[idx + stride]);
            }
        }
        
        // 交换指针
        double *temp = u;
        u = u_new;
        u_new = temp;
        grid->h_u = u;
        grid->h_u_new = u_new;
        
        // 交换虚拟边界
        exchange_ghost_mpi(grid, cfg);
        
        iter++;
        
        // 每50次迭代计算残差
        if (iter % 50 == 0) {
            resid = compute_residual_cpu(grid, cfg);
            if (rank == 0 && iter % 500 == 0) {
                printf("迭代 %d, 残差 = %.2e\n", iter, resid);
            }
        }
    }
    
    double end_time = MPI_Wtime();
    
    if (rank == 0) {
        printf("Jacobi完成! 计算时间: %.4f 秒 (迭代: %d, 残差: %.2e)\n",
               end_time - start_time, iter, resid);
    }
    
    return end_time - start_time;
}

/* ============================================================
 * Gauss-Seidel迭代（CPU版本，红黑排序）
 * ============================================================ */

double solve_gs_cpu(Grid *grid, SolverConfig *cfg) {
    int nx_local = grid->nx_local;
    int ny_local = grid->ny_local;
    int stride = grid->stride;
    double *u = grid->h_u;
    int rank = cfg->rank;
    int ny_start = cfg->ny_start;
    int iter = 0;
    double resid = cfg->tol + 1.0;
    
    double start_time = MPI_Wtime();
    
    while (resid > cfg->tol && iter < cfg->max_iters) {
        // 红黑排序：先红点后黑点
        for (int color = 0; color < 2; color++) {
            #pragma omp parallel for collapse(2)
            for (int i = 1; i <= ny_local; i++) {
                for (int j = 1; j <= nx_local; j++) {
                    int global_i = i + ny_start - 1;
                    if ((global_i + j) % 2 == color) {
                        int idx = i * stride + j;
                        u[idx] = 0.25 * (u[idx - 1] + u[idx + 1] +
                                         u[idx - stride] + u[idx + stride]);
                    }
                }
            }
            // 每更新完一种颜色，交换虚拟边界
            exchange_ghost_mpi(grid, cfg);
        }
        
        iter++;
        
        if (iter % 50 == 0) {
            resid = compute_residual_cpu(grid, cfg);
            if (rank == 0 && iter % 500 == 0) {
                printf("迭代 %d, 残差 = %.2e\n", iter, resid);
            }
        }
    }
    
    double end_time = MPI_Wtime();
    
    if (rank == 0) {
        printf("Gauss-Seidel完成! 计算时间: %.4f 秒 (迭代: %d, 残差: %.2e)\n",
               end_time - start_time, iter, resid);
    }
    
    return end_time - start_time;
}

/* ============================================================
 * SOR迭代（CPU版本，红黑排序 + 松弛因子）
 * ============================================================ */

double solve_sor_cpu(Grid *grid, SolverConfig *cfg) {
    int nx_local = grid->nx_local;
    int ny_local = grid->ny_local;
    int stride = grid->stride;
    double *u = grid->h_u;
    double omega = cfg->omega;
    int rank = cfg->rank;
    int ny_start = cfg->ny_start;
    int iter = 0;
    double resid = cfg->tol + 1.0;
    
    double start_time = MPI_Wtime();
    
    while (resid > cfg->tol && iter < cfg->max_iters) {
        // 红黑排序SOR
        for (int color = 0; color < 2; color++) {
            #pragma omp parallel for collapse(2)
            for (int i = 1; i <= ny_local; i++) {
                for (int j = 1; j <= nx_local; j++) {
                    int global_i = i + ny_start - 1;
                    if ((global_i + j) % 2 == color) {
                        int idx = i * stride + j;
                        double old_val = u[idx];
                        double relaxation = 0.25 * (u[idx - 1] + u[idx + 1] +
                                                    u[idx - stride] + u[idx + stride]);
                        u[idx] = (1.0 - omega) * old_val + omega * relaxation;
                    }
                }
            }
            exchange_ghost_mpi(grid, cfg);
        }
        
        iter++;
        
        if (iter % 50 == 0) {
            resid = compute_residual_cpu(grid, cfg);
            if (rank == 0 && iter % 500 == 0) {
                printf("迭代 %d, 残差 = %.2e\n", iter, resid);
            }
        }
    }
    
    double end_time = MPI_Wtime();
    
    if (rank == 0) {
        printf("SOR完成! 计算时间: %.4f 秒 (迭代: %d, 残差: %.2e)\n",
               end_time - start_time, iter, resid);
    }
    
    return end_time - start_time;
}

/* ============================================================
 * CPU求解器统一入口
 * ============================================================ */

double solve_cpu(Grid *grid, SolverConfig *cfg) {
    switch (cfg->method) {
        case JACOBI:
            return solve_jacobi_cpu(grid, cfg);
        case GAUSS_SEIDEL:
            return solve_gs_cpu(grid, cfg);
        case SOR:
            return solve_sor_cpu(grid, cfg);
        default:
            return solve_sor_cpu(grid, cfg);
    }
}

void free_grid(Grid *grid) {
    if (grid->h_u) free(grid->h_u);
    if (grid->h_u_new) free(grid->h_u_new);
    grid->h_u = NULL;
    grid->h_u_new = NULL;
}

void finalize_mpi() {
    MPI_Finalize();
}