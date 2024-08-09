#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <iostream>


// Error checking macro
#define CHECK_CUDA_ERROR(call) {                                          \
    cudaError_t err = call;                                               \
    if (err != cudaSuccess) {                                             \
        std::cerr << "CUDA error in file '" << __FILE__                   \
                  << "' in line " << __LINE__ << ": "                     \
                  << cudaGetErrorString(err) << " (" << err << ")"        \
                  << std::endl;                                           \
        cudaDeviceReset();                                                \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
}

#define NSPEEDS         9
#define PARAMFILE       "input.params"
#define OBSTACLEFILE    "obstacles_300x200.dat"
#define FINALSTATEFILE  "final_state%s%s.dat"
#define AVVELSFILE      "av_vels%s%s.dat"

char finalStateFile[128];
char avVelocityFile[128];

/* struct to hold the parameter values */
typedef struct {
    int nx;            /* no. of cells in y-direction */
    int ny;            /* no. of cells in x-direction */
    int maxIters;      /* no. of iterations */
    int reynolds_dim;  /* dimension for Reynolds number */
    double density;    /* density per link */
    double accel;      /* density redistribution */
    double omega;      /* relaxation parameter */
} t_param;


/* struct to hold the 'speed' values */
typedef struct {
    double speeds[NSPEEDS];
} t_speed;

void die(const char *message, const int line, const char *file) {
    fprintf(stderr, "Error at line %d of file %s:\n", line, file);
    fprintf(stderr, "%s\n", message);
    fflush(stderr);
    exit(EXIT_FAILURE);
}


int initialise(t_param *params, t_speed **cells_ptr, t_speed **tmp_cells_ptr, int **obstacles_ptr) {
    FILE *fp;          /* file pointer */
    int ii, jj;        /* generic counters */
    int xx, yy;        /* generic array indices */
    int blocked;       /* indicates whether a cell is blocked by an obstacle */
    int retval;        /* to hold return value for checking */
    double w0, w1, w2; /* weighting factors */

    /* open the parameter file */
    fp = fopen(PARAMFILE, "r");
    if (fp == NULL) {
        die("could not open file input.params", __LINE__, __FILE__);
    }

    /* read in the parameter values */
    retval = fscanf(fp, "%d\n", &(params->nx));
    if (retval != 1)
        die("could not read param file: nx", __LINE__, __FILE__);
    retval = fscanf(fp, "%d\n", &(params->ny));
    if (retval != 1)
        die("could not read param file: ny", __LINE__, __FILE__);
    retval = fscanf(fp, "%d\n", &(params->maxIters));
    if (retval != 1)
        die("could not read param file: maxIters", __LINE__, __FILE__);
    retval = fscanf(fp, "%d\n", &(params->reynolds_dim));
    if (retval != 1)
        die("could not read param file: reynolds_dim", __LINE__, __FILE__);
    retval = fscanf(fp, "%lf\n", &(params->density));
    if (retval != 1)
        die("could not read param file: density", __LINE__, __FILE__);
    retval = fscanf(fp, "%lf\n", &(params->accel));
    if (retval != 1)
        die("could not read param file: accel", __LINE__, __FILE__);
    retval = fscanf(fp, "%lf\n", &(params->omega));
    if (retval != 1)
        die("could not read param file: omega", __LINE__, __FILE__);

    /* and close up the file */
    fclose(fp);

    *cells_ptr = (t_speed *) malloc(sizeof(t_speed) * (params->ny * params->nx));
    if (*cells_ptr == NULL)
        die("cannot allocate memory for cells", __LINE__, __FILE__);

    /* the map of obstacles */
    *obstacles_ptr = (int *)malloc(sizeof(int) * (params->ny * params->nx));
    if (*obstacles_ptr == NULL)
        die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

    /* initialise densities */
    w0 = params->density * 4.0 / 9.0;
    w1 = params->density / 9.0;
    w2 = params->density / 36.0;

    for (ii = 0; ii < params->ny; ii++) {
        for (jj = 0; jj < params->nx; jj++) {
            /* centre */
            (*cells_ptr)[ii * params->nx + jj].speeds[0] = w0;
            /* axis directions */
            (*cells_ptr)[ii * params->nx + jj].speeds[1] = w1;
            (*cells_ptr)[ii * params->nx + jj].speeds[2] = w1;
            (*cells_ptr)[ii * params->nx + jj].speeds[3] = w1;
            (*cells_ptr)[ii * params->nx + jj].speeds[4] = w1;
            /* diagonals */
            (*cells_ptr)[ii * params->nx + jj].speeds[5] = w2;
            (*cells_ptr)[ii * params->nx + jj].speeds[6] = w2;
            (*cells_ptr)[ii * params->nx + jj].speeds[7] = w2;
            (*cells_ptr)[ii * params->nx + jj].speeds[8] = w2;
        }
    }

    /* first set all cells in obstacle array to zero */
    for (ii = 0; ii < params->ny; ii++) {
        for (jj = 0; jj < params->nx; jj++) {
            (*obstacles_ptr)[ii * params->nx + jj] = 0;
        }
    }

    /* open the obstacle data file */
    fp = fopen(OBSTACLEFILE, "r");
    if (fp == NULL) {
        die("could not open file obstacles", __LINE__, __FILE__);
    }

    /* read-in the blocked cells list */
    while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF) {
        /* some checks */
        if (retval != 3)
            die("expected 3 values per line in obstacle file", __LINE__, __FILE__);
        if (xx < 0 || xx > params->nx - 1)
            die("obstacle x-coord out of range", __LINE__, __FILE__);
        if (yy < 0 || yy > params->ny - 1)
            die("obstacle y-coord out of range", __LINE__, __FILE__);
        if (blocked != 1)
            die("obstacle blocked value should be 1", __LINE__, __FILE__);
        /* assign to array */
        (*obstacles_ptr)[yy * params->nx + xx] = blocked;
    }

    /* and close the file */
    fclose(fp);

    return EXIT_SUCCESS;
}



__global__ void accelerate_flow(const t_param params, t_speed *cells, const int *obstacles)
{
    int ii = threadIdx.x + blockIdx.x * blockDim.x;
    int offset; /* generic counters */
    double *speeds;

    /* compute weighting factors */
    const double w1 = params.density * params.accel / 9.0;
    const double w2 = params.density * params.accel / 36.0;

    if (ii >= params.ny) return; // NEED TO REVIEW. Might change it to ii > params.ny

    offset = ii * params.nx /* + jj (where jj=0) */;
    speeds = cells[offset].speeds;
    /* if the cell is not occupied and we don't send a density negative */
    if (!obstacles[offset] && (speeds[3] - w1) > 0.0 && (speeds[6] - w2) > 0.0 && (speeds[7] - w2) > 0.0) {
        /* increase 'east-side' densities */
        speeds[1] += w1;
        speeds[5] += w2;
        speeds[8] += w2;
        /* decrease 'west-side' densities */
        speeds[3] -= w1;
        speeds[6] -= w2;
        speeds[7] -= w2;
    }
}

__global__ void collision(const t_param params, t_speed *src_cells, t_speed *dst_cells, const int *obstacles)
{
    int ii = blockIdx.y * blockDim.y + threadIdx.y;
    int jj = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = ii * params.nx + jj;

    if (ii >= params.ny || jj >= params.nx) return;

    int kk;
    const double w0 = 4.0 / 9.0;       /* weighting factor */
    const double w1 = 1.0 / 9.0;       /* weighting factor */
    const double w2 = 1.0 / 36.0;      /* weighting factor */
    double u_x, u_y;                   /* av. velocities in x and y directions */
    double u[NSPEEDS];                 /* directional velocities */
    double d_equ[NSPEEDS];             /* equilibrium densities */
    double u_sq;                       /* squared velocity */
    double local_density;              /* sum of densities in a particular cell */
    int x_e, x_w, y_n, y_s;            /* indices of neighbouring cells */
    double speeds[NSPEEDS];

    double *dst_speeds = dst_cells[offset].speeds;

    /* PROPAGATE: determine indices of axis-direction neighbours respecting periodic boundary conditions (wrap around) */
        y_s = (ii + 1) % params.ny;
        x_w = (jj + 1) % params.nx;
        y_n = (ii == 0) ? (ii + params.ny - 1) : (ii - 1);
        x_e = (jj == 0) ? (jj + params.nx - 1) : (jj - 1);

        /* if the cell contains an obstacle */
        if (obstacles[ii * params.nx + jj])
        {
            /* PROPAGATE ANB REBOUND: propagate and mirror densities from neighbouring cells into the current cell. */
            dst_speeds[0] = src_cells[ii * params.nx + jj].speeds[0];  /* central cell, no movement */
            dst_speeds[1] = src_cells[ii * params.nx + x_w].speeds[3];  /* west */
            dst_speeds[2] = src_cells[y_s * params.nx + jj].speeds[4];  /* south */
            dst_speeds[3] = src_cells[ii * params.nx + x_e].speeds[1];  /* east */
            dst_speeds[4] = src_cells[y_n * params.nx + jj].speeds[2];  /* north */
            dst_speeds[5] = src_cells[y_s * params.nx + x_w].speeds[7];  /* south-west */
            dst_speeds[6] = src_cells[y_s * params.nx + x_e].speeds[8];  /* south-east */
            dst_speeds[7] = src_cells[y_n * params.nx + x_e].speeds[5];  /* north-east */
            dst_speeds[8] = src_cells[y_n * params.nx + x_w].speeds[6];  /* north-west */
        }

            else
            {
                /* PROPAGATE: propagate densities from neighbouring cells into the current cells, following appropriate directions of
                 * travel and writing into a temporary buffer.
                 */
                speeds[0] = src_cells[ii * params.nx + jj].speeds[0];  /* central cell, no movement */
                speeds[1] = src_cells[ii * params.nx + x_e].speeds[1];  /* east */
                speeds[2] = src_cells[y_n * params.nx + jj].speeds[2];  /* north */
                speeds[3] = src_cells[ii * params.nx + x_w].speeds[3];  /* west */
                speeds[4] = src_cells[y_s * params.nx + jj].speeds[4];  /* south */
                speeds[5] = src_cells[y_n * params.nx + x_e].speeds[5];  /* north-east */
                speeds[6] = src_cells[y_n * params.nx + x_w].speeds[6];  /* north-west */
                speeds[7] = src_cells[y_s * params.nx + x_w].speeds[7];  /* south-west */
                speeds[8] = src_cells[y_s * params.nx + x_e].speeds[8];  /* south-east */

                /* COLLISION */
                /* compute local density total */
                local_density = speeds[0] + speeds[1] + speeds[2] + speeds[3] + speeds[4] + speeds[5] + speeds[6] + speeds[7] + speeds[8];
                
                /* compute x velocity component */
                u_x = (speeds[1] + speeds[5] + speeds[8] - (speeds[3] + speeds[6] + speeds[7])) / local_density;
                
                /* compute y velocity component */
                u_y = (speeds[2] + speeds[5] + speeds[6] - (speeds[4] + speeds[7] + speeds[8])) / local_density;
               
                /* directional velocity components */
                u[1] = u_x;       /* east */
                u[2] = u_y;       /* north */
                u[5] = u_x + u_y; /* north-east */
                u[6] = -u_x + u_y; /* north-west */

                /* velocity squared over twice the speed of sound */
                u_sq = (u_x * u_x + u_y * u_y) * 1.5;
                /* equilibrium densities */
                /* zero velocity density: weight w0 */
                d_equ[0] = w0 * local_density * (1.0 - u_sq);

                /* axis speeds: weight w1 */
                d_equ[1] = w1 * local_density * (1.0 + u[1] * 3.0 + (u[1] * u[1] * 4.5) - u_sq);
                d_equ[2] = w1 * local_density * (1.0 + u[2] * 3.0 + (u[2] * u[2] * 4.5) - u_sq);
                d_equ[3] = w1 * local_density * (1.0 - u[1] * 3.0 + (u[1] * u[1] * 4.5) - u_sq);
                d_equ[4] = w1 * local_density * (1.0 - u[2] * 3.0 + (u[2] * u[2] * 4.5) - u_sq);
                /* diagonal speeds: weight w2 */
                d_equ[5] = w2 * local_density * (1.0 + u[5] * 3.0 + (u[5] * u[5] * 4.5) - u_sq);
                d_equ[6] = w2 * local_density * (1.0 + u[6] * 3.0 + (u[6] * u[6] * 4.5) - u_sq);
                d_equ[7] = w2 * local_density * (1.0 - u[5] * 3.0 + (u[5] * u[5] * 4.5) - u_sq);
                d_equ[8] = w2 * local_density * (1.0 - u[6] * 3.0 + (u[6] * u[6] * 4.5) - u_sq);

                /* relaxation step */
                for (kk = 0; kk < NSPEEDS; kk++) {
                    speeds[kk] += params.omega * (d_equ[kk] - speeds[kk]);
                }
                *((t_speed *) dst_speeds) = *((t_speed *) speeds);

            }
}


__global__ void compute_av_velocity_kernel(const t_param params, t_speed* d_cells, int* d_obstacles, double* d_tot_u_x, int* d_tot_cells) {
    int ii = blockIdx.y * blockDim.y + threadIdx.y;
    int jj = blockIdx.x * blockDim.x + threadIdx.x;

    if (ii < params.ny && jj < params.nx) {
        int offset = ii * params.nx + jj;

        if (!d_obstacles[offset]) {
            t_speed cell = d_cells[offset];
            double local_density = cell.speeds[0] + cell.speeds[1] + cell.speeds[2] + cell.speeds[3] + cell.speeds[4] + cell.speeds[5] + cell.speeds[6] + cell.speeds[7] + cell.speeds[8];
            double u_x = (cell.speeds[1] + cell.speeds[5] + cell.speeds[8] - (cell.speeds[3] + cell.speeds[6] + cell.speeds[7])) / local_density;

            atomicAdd(d_tot_u_x, u_x);
            atomicAdd(d_tot_cells, 1);
        }
    }
}

double av_velocity(const t_param params, t_speed* cells, int* obstacles) {
    t_speed* d_cells;
    int* d_obstacles;
    double* d_tot_u_x;
    int* d_tot_cells;

    size_t cells_size = params.nx * params.ny * sizeof(t_speed);
    size_t obstacles_size = params.nx * params.ny * sizeof(int);

    // Allocate device memory
    cudaMalloc(&d_cells, cells_size);
    cudaMalloc(&d_obstacles, obstacles_size);
    cudaMalloc(&d_tot_u_x, sizeof(double));
    cudaMalloc(&d_tot_cells, sizeof(int));

    // Initialize device memory
    cudaMemset(d_tot_u_x, 0, sizeof(double));
    cudaMemset(d_tot_cells, 0, sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_cells, cells, cells_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_obstacles, obstacles, obstacles_size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((params.nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (params.ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    compute_av_velocity_kernel<<<numBlocks, threadsPerBlock>>>(params, d_cells, d_obstacles, d_tot_u_x, d_tot_cells);

    // Copy results back to host
    double tot_u_x;
    int tot_cells;
    cudaMemcpy(&tot_u_x, d_tot_u_x, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&tot_cells, d_tot_cells, sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_cells);
    cudaFree(d_obstacles);
    cudaFree(d_tot_u_x);
    cudaFree(d_tot_cells);

    return tot_cells > 0 ? tot_u_x / (double) tot_cells : 0.0;
}


__host__ double calculate_reynolds_number(const t_param params, double avg_velocity) {
    // The formula for Reynolds number
    double nu = 1.0 / 6.0 * (2.0 / params.omega - 1.0);  // Kinematic viscosity
    return (avg_velocity * params.reynolds_dim) / nu;
}


int main(int argc, char* argv[])
{
    t_param params;            /* struct to hold parameter values */
    //src_cells needs cudaMemcpy
    t_speed *src_cells = NULL; /* source grid containing fluid densities */
    //dst and tmp needs cudaMalloc
    t_speed *dst_cells = NULL; /* destination grid containing fluid densities */
    t_speed *temp_swap = NULL; /* temporary cell pointer variable used to swap source and destination grid pointers */
    // obstacles needs cudaMemcpy
    int *obstacles = NULL;     /* grid indicating which cells are blocked */
    // av_vels needs cudaMalloc
    double av_vels = 0.0; 
    int ii;                    /* generic counter */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;


    if (argc > 1) {
        sprintf(finalStateFile, FINALSTATEFILE, ".", argv[1]);
        sprintf(avVelocityFile, AVVELSFILE, ".", argv[1]);
    } else {
        sprintf(finalStateFile, FINALSTATEFILE, "", "");
        sprintf(avVelocityFile, AVVELSFILE, "", "");
    }

    /* initialise our data structures and load values from file */
    initialise(&params, &src_cells, &dst_cells, &obstacles);

    t_param* device_params;
    CHECK_CUDA_ERROR(cudaMalloc(&device_params, sizeof(t_param)));
    CHECK_CUDA_ERROR(cudaMemcpy(device_params, &params, sizeof(t_param), cudaMemcpyHostToDevice));

    t_speed* device_src_cells;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&device_src_cells, sizeof(t_speed) * (params.ny * params.nx)));
    CHECK_CUDA_ERROR(cudaMemcpy(device_src_cells, src_cells, sizeof(t_speed) * (params.ny * params.nx), cudaMemcpyHostToDevice));

    t_speed* device_dst_cells;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&device_dst_cells, sizeof(t_speed) * (params.ny * params.nx)));
    
    int* device_obstacles;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&device_obstacles, sizeof(int) * (params.ny * params.nx)));
    CHECK_CUDA_ERROR(cudaMemcpy(device_obstacles, obstacles, sizeof(int) * (params.ny * params.nx), cudaMemcpyHostToDevice));

    t_speed* device_temp_swap;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&device_temp_swap, sizeof(t_speed)));

    dim3 blockDim(16);
    dim3 gridDim((params.ny + blockDim.y - 1) / blockDim.y);

    dim3 blockDim2(16,16);
    dim3 gridDim2((params.nx + blockDim.x - 1) / blockDim.x, (params.ny + blockDim.y - 1) / blockDim.y);

    cudaEventRecord(start);

    for (ii = 0; ii < params.maxIters; ii++)
    {
        accelerate_flow<<<gridDim,blockDim>>>(params, device_src_cells, device_obstacles);
        cudaDeviceSynchronize();
        collision<<<gridDim2,blockDim2>>>(params, device_src_cells, device_dst_cells, device_obstacles);
        cudaDeviceSynchronize();
        temp_swap = device_src_cells;
        device_src_cells = device_dst_cells;
        device_dst_cells = temp_swap;
    }

    CHECK_CUDA_ERROR(cudaMemcpy(src_cells, device_src_cells, sizeof(t_speed) * (params.ny * params.nx), cudaMemcpyDeviceToHost));
    
    av_vels = av_velocity(params, src_cells, obstacles);

    // Record the stop event
    cudaEventRecord(stop);

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);

    double reynolds_number = calculate_reynolds_number(params, av_vels);
    printf("Reynolds number:\t%.12E\n", reynolds_number);
    printf("Time elapsed: %f ms\n", milliseconds);

}