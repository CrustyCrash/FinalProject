# Boltzmann Fluid Simulation (OpenACC Version)

==============================================

## Overview

This project is a Boltzmann Fluid Simulation implemented using OpenACC for parallel computing. This version is specifically tailored for Linux systems.

## Pre-requisites

To build and run the simulation, you need a compiler that supports OpenACC. Options include:

- **NVIDIA HPC SDK**: Install it by following instructions on [NVIDIA's website](https://developer.nvidia.com/hpc-sdk).
- **NVIDIA HPC Compiler**: Formerly known as PGI Compiler, installation instructions are available on [NVIDIA's website](https://developer.nvidia.com/hpc-sdk).

Make sure you have `cmake` and `make` installed. You can install them using your package manager. For example, on Debian-based systems:

```shell
sudo apt-get update
sudo apt-get install cmake make
re