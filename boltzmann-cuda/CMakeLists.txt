cmake_minimum_required(VERSION 3.14.5)
project(boltzmann-cuda)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Enable CUDA
find_package(CUDA REQUIRED)

# Set the CUDA architecture flags for RTX 3060
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_86,code=sm_86")

# Add the executable
cuda_add_executable(boltzmann-cuda boltzmann-cuda.cu input.params)

option(DEBUG "Enable debug mode" OFF)
if (DEBUG)
    message("-- Debug mode enabled")
    add_definitions(-DDEBUG)
else (DEBUG)
    message("-- Debug mode disabled")
endif (DEBUG)

# Link CUDA libraries
target_link_libraries(boltzmann-cuda ${CUDA_LIBRARIES})

# Configure files to be copied to the build directory
configure_file(input.params ${CMAKE_CURRENT_BINARY_DIR}/input.params COPYONLY)
configure_file(obstacles.dat ${CMAKE_CURRENT_BINARY_DIR}/obstacles.dat COPYONLY)
configure_file(obstacles_300x200.dat ${CMAKE_CURRENT_BINARY_DIR}/obstacles_300x200.dat COPYONLY)
