# CUDA-Accelerated Lattice Boltzmann Method (LBM) B2Q9 Code
This repository contains a CUDA-accelerated implementation of the Lattice Boltzmann Method (LBM) using the B2Q9 model. The code is designed to simulate fluid dynamics and calculate the Reynolds number for various flow configurations.

## Table of Contents
1)  Introduction
2)  Features
3)  Prerequisites
4)  Installation
5)  Usage
6)  Code Structure
7)  Performance
8) Contributing

## Introduction
The Lattice Boltzmann Method (LBM) is a computational fluid dynamics (CFD) technique for simulating fluid flow. The B2Q9 model is a popular 2D LBM model that uses a 9-velocity lattice structure. This implementation leverages the power of NVIDIA GPUs using CUDA to accelerate the computation, making it significantly faster than traditional CPU-based implementations.

## Features
1)  CUDA Acceleration: Speeds up the LBM computation using parallel processing on GPUs.
2)  B2Q9 Model: Implements the standard 9-velocity lattice model.
3)  Reynolds Number Calculation: Computes the Reynolds number for different flow conditions.
4)  Scalability: Designed to handle large-scale simulations efficiently.

## Prerequisites
To compile and run this code, you will need the following:

1)  CUDA Toolkit: Ensure that you have the appropriate version of the CUDA Toolkit installed.
2)  NVIDIA GPU: A CUDA-capable NVIDIA GPU is required to run the accelerated code.
3)  C++ Compiler: A C++ compiler that supports CUDA (e.g., nvcc).
4)  CMake
5)  Make
6)  MSVC (to build project with windows)

## Installation
Clone the repository:

```sh
git clone <repo-name>
cd <cloned-dir>
cmake .
make
```
For Windows
```powershell
git clone <repo-name>
cd <cloned-repo>
cmake .
## open .sln in MSVC and build the project
```

## Usage
```sh
./boltzmann-cuda
```
Windows
```powershell
# run the solution from MSVC, the executable is found in the debug folder
```
## Code Structure
1)  boltzmann-cuda.cu: The main CUDA-accelerated LBM B2Q9 code.
2)  CMakeLists.txt
3)  input.params
4)  obstacles.dat
5)  obstacles_300x200.dat
6)  README.md: Documentation and usage instructions.

## Performance
The CUDA implementation of the LBM B2Q9 model significantly reduces the computation time compared to serial CPU-based implementations. Benchmarks and performance comparisons can be found in the 'profiling' folder.

## Contributing
Contributions are welcome! Please fork this repository, create a new branch, and submit a pull request with your changes.

