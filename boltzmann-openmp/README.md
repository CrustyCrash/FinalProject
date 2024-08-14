D2Q9-BGK Lattice Boltzmann Scheme
Introduction
This project implements a D2Q9-BGK lattice Boltzmann method, a numerical simulation approach used to model fluid dynamics. The scheme operates on a 2D grid where each cell interacts with its neighbors through nine discrete velocity vectors. The Bhatnagar-Gross-Krook (BGK) model is employed to simulate the collision step within the lattice.

Key Concepts:
D2Q9: Refers to a 2D grid with 9 velocity directions per grid cell.
BGK: A collision model that relaxes the system towards equilibrium.
The simulation computes the flow characteristics such as velocity and Reynolds number, providing insights into fluid behavior under various conditions.

Table of Contents
Introduction
Installation
Usage
Features
Profiling Results
License
Installation
To run the simulation, ensure you have the following dependencies installed:

A C compiler (e.g., GCC)
Profiling tools (e.g., Gprof, VTune)
CMake
Make (if a Makefile is provided)
Clone the repository and compile the code:

bash
Copy code
git clone <repository-url>
cd <repository-directory>
cmake .
make

For Windows:
powershell/ git bash
git clone <repository-url>
cd <repository-directory>
cmake .
build project using msvc

Usage
Run the simulation using the following command:

bash
Copy code
./lattice_boltzmann
The output will include the Reynolds number, CPU and wall clock time, and details about the number of threads used during execution.

Example Output:
text
Copy code
Reynolds number:        2.641710388877E+01
Elapsed CPU time:       121036 (ms)
Elapsed wall time:      6060 (ms)
Using default number of threads: 20 threads
The results demonstrate parallelism in the simulation, as indicated by the difference between CPU time and wall time.

Features
2D Grid Simulation: Implements a D2Q9 lattice Boltzmann model on a 2D grid.
BGK Collision Model: Utilizes the Bhatnagar-Gross-Krook model for collision steps.
Parallel Computing Support: Leverages OpenMP for parallel execution, enhancing computational efficiency and enabling large-scale simulations.
Adaptive Grid Resolution: Allows customization of grid resolution, providing flexibility for different simulation scales
Continuous Integration Support: Includes a Jenkins pipeline configuration for automated building, testing, and deployment.
Performance Tuning: Provides tools and options for fine-tuning the simulation performance, including setting the number of threads and optimizing cache usage.
Reynolds Number Calculation: Automatically computes the Reynolds number to characterize the flow regime.



VTune Profiling
The VTune profiling results are awaited and will provide additional insights into the performance and parallelism of the code.

License
This project is licensed under the GNU General Public License (GPL). You are free to copy, modify, and distribute this software as long as the original copyright notice is preserved.
