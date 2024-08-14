cmake_minimum_required(VERSION 3.14.5)
project(boltzmann-openacc)

set(CMAKE_CXX_STANDARD 17)

# Enable OpenACC
if (CMAKE_C_COMPILER_ID MATCHES "GNU")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenacc")
elseif (CMAKE_C_COMPILER_ID MATCHES "Cray")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -hacc")
elseif (CMAKE_C_COMPILER_ID MATCHES "PGI")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -acc")
elseif (CMAKE_C_COMPILER_ID MATCHES "Intel")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -qopenacc")
endif()

# Add executable
add_executable(boltzmann-openacc b2q9_openACC.c)