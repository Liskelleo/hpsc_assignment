#ifndef LAPLACE_SOLVER_H
#define LAPLACE_SOLVER_H

#include <cuda_runtime.h>
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* ============================================================
 * 迭代方法枚举
 * ============================================================ */
typedef enum {
    JACOBI,        // Jacobi迭代
    GAUSS_SEIDEL,  // Gauss-Seidel迭代
    SOR            // 逐次超松弛迭代
} IterMethod;

/* ============================================================
 * 求解器配置结构体
 * ============================================================ */
typedef struct {
    int nx, ny;              // 全局网格内点数（不含边界）
    int nx_local, ny_local;  // 本地网格内点数（不含边界）
    int nx_start, ny_start;  // 本地网格在全局中的起始索引
    double dx, dy;           // 网格步长
    double tol;              // 收敛容差
    int max_iters;           // 最大迭代次数
    double omega;            // SOR松弛因子
    IterMethod method;       // 迭代方法
    int rank, size;          // MPI进程信息
} SolverConfig;

/* ============================================================
 * 网格数据结构体（包含ghost cells）
 * ============================================================ */
typedef struct {
    double *h_u;             // 主机端数组（一维连续，大小 (ny_local+2)*(nx_local+2)）
    double *h_u_new;         // 主机端新值数组
    double *d_u;             // 设备端数组
    double *d_u_new;         // 设备端新值数组
    double *h_pin_row_snd;   // pinned host row send buffer
    double *h_pin_row_rcv;   // pinned host row recv buffer
    double *h_pin_col_snd;   // pinned host column send buffer
    double *h_pin_col_rcv;   // pinned host column recv buffer
    double *d_col_l;         // device column buffer, left
    double *d_col_r;         // device column buffer, right
    int stride;              // 行步长 = nx_local + 2
    int ny_local;            // 本地行数
    int nx_local;            // 本地列数
} Grid;

/* ============================================================
 * 函数声明
 * ============================================================ */

// 初始化
void init_mpi(int *argc, char ***argv, int *rank, int *size);
void get_decomp_1d(int ny, int rank, int size, int *ny_local, int *ny_start);
void init_config(SolverConfig *cfg, int nx, int ny, double tol, int max_iters, 
                 IterMethod method, double omega, int rank, int size);
void init_grid(Grid *grid, int nx_local, int ny_local);
void init_boundary(Grid *grid, SolverConfig *cfg);
void set_initial_condition(Grid *grid, SolverConfig *cfg);

// 通信
void exchange_ghost_mpi(Grid *grid, SolverConfig *cfg);
void exchange_ghost_cuda(Grid *grid, SolverConfig *cfg, cudaStream_t stream);

// 求解器（CPU版本）
double solve_jacobi_cpu(Grid *grid, SolverConfig *cfg);
double solve_gs_cpu(Grid *grid, SolverConfig *cfg);
double solve_sor_cpu(Grid *grid, SolverConfig *cfg);
double solve_cpu(Grid *grid, SolverConfig *cfg);

// 求解器（CUDA版本）
void solve_jacobi_cuda(Grid *grid, SolverConfig *cfg, cudaStream_t stream);
void solve_gs_cuda(Grid *grid, SolverConfig *cfg, cudaStream_t stream);
void solve_sor_cuda(Grid *grid, SolverConfig *cfg, cudaStream_t stream);
double solve_cuda(Grid *grid, SolverConfig *cfg, cudaStream_t stream);

// MPI并行IO
void mpi_io_write(const char *filename, Grid *grid, SolverConfig *cfg, int iter);
void mpi_io_write_async(const char *filename, Grid *grid, SolverConfig *cfg, 
                        int iter, cudaStream_t stream);

// 工具函数
double compute_residual_cpu(Grid *grid, SolverConfig *cfg);
double compute_residual_cuda(Grid *grid, SolverConfig *cfg, cudaStream_t stream);
void print_grid(Grid *grid, SolverConfig *cfg);
void free_grid(Grid *grid);
void finalize_mpi();

#endif // LAPLACE_SOLVER_H
