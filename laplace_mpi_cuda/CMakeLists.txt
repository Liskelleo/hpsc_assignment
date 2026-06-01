# cmake_minimum_required(VERSION 3.12)
# project(LaplaceMPICUDA LANGUAGES CXX CUDA)

# # 设置C++标准
# set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # 开启CUDA支持
# find_package(CUDA REQUIRED)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_61")

# # 查找MPI
# find_package(MPI REQUIRED)

# # 查找OpenMP
# find_package(OpenMP REQUIRED)

# # 设置源文件
# set(SOURCES
#     src/laplace_solver.cpp
#     src/laplace_cuda.cu
#     src/mpi_io.cpp
# )

# set(HEADERS
#     src/laplace_solver.h
# )

# # 生成可执行文件
# add_executable(laplace_mpi_cuda ${SOURCES} ${HEADERS})

# # 设置包含目录
# target_include_directories(laplace_mpi_cuda PRIVATE ${CMAKE_SOURCE_DIR}/src)

# # 链接库
# target_link_libraries(laplace_mpi_cuda
#     ${CUDA_LIBRARIES}
#     ${MPI_CXX_LIBRARIES}
#     OpenMP::OpenMP_CXX
# )

# # 设置编译标志
# if(MPI_CXX_COMPILE_FLAGS)
#     set_target_properties(laplace_mpi_cuda PROPERTIES
#         COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
# endif()

# # 可选：开启异步IO支持（需要添加相应的标志）
# # target_compile_definitions(laplace_mpi_cuda PRIVATE USE_ASYNC_IO)

cmake_minimum_required(VERSION 3.16)
project(laplace_mpi_cuda LANGUAGES CXX CUDA)

# 寻找 MPI 和 经典 CUDA 包
find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

# 添加你的可执行文件
add_executable(laplace_mpi_cuda 
    src/laplace_solver.cpp
    src/laplace_cuda.cu
    src/mpi_io.cpp
    src/stream_async.cu
    src/main_async.cu
)

# 喂给编译器正确的头文件路径
target_include_directories(laplace_mpi_cuda PRIVATE 
    ${CUDA_INCLUDE_DIRS} 
    ${MPI_CXX_INCLUDE_PATH}
)

# 强行指定 CXX 标准为 C++17
set_target_properties(laplace_mpi_cuda PROPERTIES CXX_STANDARD 17)

# 绕过 CMake 的标准检查，直接用 flags 告诉 nvcc 和 g++ 使用 c++17
target_compile_options(laplace_mpi_cuda PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>
    $<$<COMPILE_LANGUAGE:CUDA>:-std=c++17>
)

# 链接 MPI 和 经典 CUDA 库
target_link_libraries(laplace_mpi_cuda PRIVATE 
    MPI::MPI_CXX 
    ${CUDA_LIBRARIES}
)