/**
 * main_single_cuda.cu - single-GPU global CUDA Laplace solver.
 *
 * This version runs the whole (nx+2) x (ny+2) grid on one GPU without MPI
 * decomposition or halo exchange.  It is used as the single-card reference
 * for size scaling.
 */

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

__device__ inline double atomicMaxDoubleSingle(double *address, double val) {
    unsigned long long int *address_as_ull =
        reinterpret_cast<unsigned long long int *>(address);
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void sor_single_kernel(double *u, int nx, int ny, int stride,
                                  int color, double omega) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i <= ny && j <= nx && ((i + j) & 1) == color) {
        int idx = i * stride + j;
        double old_val = u[idx];
        double relax = 0.25 * (u[idx - 1] + u[idx + 1] +
                               u[idx - stride] + u[idx + stride]);
        u[idx] = (1.0 - omega) * old_val + omega * relax;
    }
}

__global__ void residual_single_kernel(const double *u, double *residual,
                                       int nx, int ny, int stride) {
    __shared__ double shared[256];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;

    double local = 0.0;
    if (i <= ny && j <= nx) {
        int idx = i * stride + j;
        double relax = 0.25 * (u[idx - 1] + u[idx + 1] +
                               u[idx - stride] + u[idx + stride]);
        local = fabs(relax - u[idx]);
        if (!isfinite(local)) {
            local = INFINITY;
        }
    }

    shared[tid] = local;
    __syncthreads();

    int block_threads = blockDim.x * blockDim.y;
    for (int s = block_threads / 2; s > 0; s >>= 1) {
        if (tid < s && shared[tid] < shared[tid + s]) {
            shared[tid] = shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMaxDoubleSingle(residual, shared[0]);
    }
}

static void check_cuda(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

int main(int argc, char **argv) {
    int nx = argc > 1 ? std::atoi(argv[1]) : 512;
    int ny = argc > 2 ? std::atoi(argv[2]) : 512;
    double tol = argc > 3 ? std::atof(argv[3]) : 1e-6;
    int max_iters = argc > 4 ? std::atoi(argv[4]) : 50000;
    mkdir("bin", 0755);
    std::string default_output =
        "bin/result_single_cuda_" + std::to_string(nx) + "x" + std::to_string(ny) + ".bin";
    const char *output = argc > 5 ? argv[5] : default_output.c_str();

    int stride = nx + 2;
    size_t count = static_cast<size_t>(ny + 2) * stride;
    std::vector<double> h_u(count, 0.0);
    for (int j = 1; j <= nx; ++j) {
        h_u[(ny + 1) * stride + j] = 1.0;
    }

    double omega = 2.0 / (1.0 + std::sin(M_PI / (std::max(nx, ny) + 1.0)));

    double *d_u = nullptr;
    double *d_residual = nullptr;
    check_cuda(cudaMalloc(&d_u, count * sizeof(double)), "cudaMalloc d_u");
    check_cuda(cudaMalloc(&d_residual, sizeof(double)), "cudaMalloc d_residual");
    check_cuda(cudaMemcpy(d_u, h_u.data(), count * sizeof(double),
                          cudaMemcpyHostToDevice), "initial H2D");

    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");
    check_cuda(cudaEventRecord(start), "cudaEventRecord start");

    int iter = 0;
    double resid = tol + 1.0;
    while (resid > tol && iter < max_iters) {
        sor_single_kernel<<<grid, block>>>(d_u, nx, ny, stride, 0, omega);
        sor_single_kernel<<<grid, block>>>(d_u, nx, ny, stride, 1, omega);
        ++iter;

        if (iter % 50 == 0) {
            check_cuda(cudaMemset(d_residual, 0, sizeof(double)), "residual memset");
            residual_single_kernel<<<grid, block>>>(d_u, d_residual, nx, ny, stride);
            check_cuda(cudaMemcpy(&resid, d_residual, sizeof(double),
                                  cudaMemcpyDeviceToHost), "residual D2H");
            if (iter % 500 == 0) {
                std::printf("迭代 %d, 残差 = %.2e\n", iter, resid);
            }
        }
    }

    check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");

    check_cuda(cudaMemcpy(h_u.data(), d_u, count * sizeof(double),
                          cudaMemcpyDeviceToHost), "final D2H");

    std::ofstream ofs(output, std::ios::binary);
    ofs.write(reinterpret_cast<const char *>(h_u.data()),
              static_cast<std::streamsize>(count * sizeof(double)));
    ofs.close();

    std::printf("Single-GPU CUDA完成! 计算时间: %.4f 秒 (迭代: %d, 残差: %.2e)\n",
                elapsed_ms / 1000.0f, iter, resid);
    std::printf("输出文件: %s\n", output);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_u);
    cudaFree(d_residual);
    return 0;
}
