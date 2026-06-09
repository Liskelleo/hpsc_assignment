/**
 * mpi_io.cpp - MPI parallel binary output.
 *
 * The output file is a raw double array of shape (ny + 2, nx + 2) in global
 * row-major order.  Local ghost cells are not written.  Physical boundary
 * rows are written only by the ranks that own the adjacent physical subdomain.
 */

#include "laplace_solver.h"
#include <mpi.h>
#include <vector>

void mpi_io_write(const char *filename, Grid *grid, SolverConfig *cfg, int iter) {
    (void)iter;

    const int rank = cfg->rank;
    const int size = cfg->size;
    const int nx = cfg->nx;
    const int ny = cfg->ny;
    const int nx_global = nx + 2;
    const int stride = grid->stride;
    double *u = grid->h_u;

    MPI_File fh;
    MPI_Status status;

    MPI_File_open(MPI_COMM_WORLD, filename,
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &fh);

    MPI_File_set_size(fh, (MPI_Offset)(ny + 2) * nx_global * sizeof(double));

    int first_global_row = cfg->ny_start;
    int last_global_row = cfg->ny_start + cfg->ny_local - 1;

    if (rank == 0) {
        first_global_row = 0;
    }
    if (rank == size - 1) {
        last_global_row = ny + 1;
    }

    int local_rows = last_global_row - first_global_row + 1;
    std::vector<double> packed((size_t)local_rows * nx_global);

    for (int global_i = first_global_row; global_i <= last_global_row; ++global_i) {
        int local_i;
        if (global_i == 0 && rank == 0) {
            local_i = 0;
        } else if (global_i == ny + 1 && rank == size - 1) {
            local_i = grid->ny_local + 1;
        } else {
            local_i = global_i - (cfg->ny_start - 1);
        }

        int packed_i = global_i - first_global_row;
        for (int j = 0; j < nx_global; ++j) {
            packed[(size_t)packed_i * nx_global + j] = u[local_i * stride + j];
        }
    }

    MPI_Offset offset = ((MPI_Offset)first_global_row * nx_global) * sizeof(double);
    MPI_File_write_at_all(fh, offset, packed.data(),
                          local_rows * nx_global, MPI_DOUBLE, &status);

    MPI_File_close(&fh);
}

void mpi_io_write_async(const char *filename, Grid *grid, SolverConfig *cfg,
                        int iter, cudaStream_t stream) {
    cudaStreamSynchronize(stream);
    mpi_io_write(filename, grid, cfg, iter);
}
