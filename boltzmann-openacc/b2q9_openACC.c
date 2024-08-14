#include <openacc.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef _WIN32
#include <sys/time.h>
#endif

#ifdef _WIN32
#define NULL    ((void *)0)
#endif

#define NSPEEDS         9
#define PARAMFILE       "input.params"
#define OBSTACLEFILE    "obstacles_300x200.dat"
#define FINALSTATEFILE  "final_state%s%s.dat"
#define AVVELSFILE      "av_vels%s%s.dat"

char finalStateFile[128];
char avVelocityFile[128];

typedef struct {
    int nx;
    int ny;
    int maxIters;
    int reynolds_dim;
    double density;
    double accel;
    double omega;
} t_param;

typedef struct {
    double speeds[NSPEEDS];
} t_speed;

int initialise(t_param *params, t_speed **cells_ptr, t_speed **tmp_cells_ptr, int **obstacles_ptr, double **av_vels_ptr);
int timestep(t_param params, t_speed *src_cells, t_speed *dst_cells, const int *obstacles);
int accelerate_flow(t_param params, t_speed *cells, const int *obstacles);
int collision(t_param params, t_speed *dst_cells, t_speed *src_cells, const int *obstacles);
int write_values(t_param params, t_speed *cells, int *obstacles, double *av_vels);
int finalise(t_speed **cells_ptr, t_speed **tmp_cells_ptr, int **obstacles_ptr, double **av_vels_ptr);
double total_density(t_param params, t_speed *cells);
double av_velocity(t_param params, t_speed *cells, const int *obstacles);
double calc_reynolds(t_param params, t_speed *cells, int *obstacles);
void die(const char *message, const int line, const char *file);

int main(int argc, char *argv[]) {
    t_param params;
    t_speed *src_cells = NULL;
    t_speed *dst_cells = NULL;
    t_speed *temp_swap = NULL;
    int *obstacles = NULL;
    double *av_vels = NULL;
    int ii;
    clock_t tic, toc;

    printf("Running Boltzman OpenACC simulation...\n");

#ifndef _WIN32
    struct timeval tv1, tv2, tv3;
#endif

    if (argc > 1) {
        sprintf(finalStateFile, FINALSTATEFILE, ".", argv[1]);
        sprintf(avVelocityFile, AVVELSFILE, ".", argv[1]);
    } else {
        sprintf(finalStateFile, FINALSTATEFILE, "", "");
        sprintf(avVelocityFile, AVVELSFILE, "", "");
    }

    initialise(&params, &src_cells, &dst_cells, &obstacles, &av_vels);

    tic = clock();
#ifndef _WIN32
    gettimeofday(&tv1, NULL);
#endif

    #pragma acc enter data copyin(params) copyout(src_cells[:params.nx*params.ny], dst_cells[:params.nx*params.ny], obstacles[:params.nx*params.ny], av_vels[:params.maxIters])
    for (ii = 0; ii < params.maxIters; ii++) {
        timestep(params, src_cells, dst_cells, obstacles);
        temp_swap = src_cells;
        src_cells = dst_cells;
        dst_cells = temp_swap;
        av_vels[ii] = av_velocity(params, src_cells, obstacles);
#ifdef DEBUG
        printf("==timestep: %d==\n",ii);
        printf("av velocity: %.12E\n", av_vels[ii]);
        printf("tot density: %.12E\n",total_density(params,src_cells));
#endif
    }

#ifndef _WIN32
    gettimeofday(&tv2, NULL);
    timersub(&tv2, &tv1, &tv3);
#endif
    toc = clock();

    printf("==done==\n");
    printf("Reynolds number:\t%.12E\n", calc_reynolds(params, src_cells, obstacles));
    printf("Elapsed CPU time:\t%ld (ms)\n", (toc - tic) / (CLOCKS_PER_SEC / 1000));
#ifndef _WIN32
    printf("Elapsed wall time:\t%ld (ms)\n", (tv3.tv_sec * 1000) + (tv3.tv_usec / 1000));
#endif
#ifdef _OPENACC
    printf("Threads used:\t\t%d\n", acc_get_num_devices(acc_device_nvidia));
#endif
    write_values(params, src_cells, obstacles, av_vels);
    finalise(&src_cells, &dst_cells, &obstacles, &av_vels);

    return EXIT_SUCCESS;
}

int timestep(const t_param params, t_speed *src_cells, t_speed *dst_cells, const int *obstacles) {
    accelerate_flow(params, src_cells, obstacles);
    collision(params, src_cells, dst_cells, obstacles);
    return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, t_speed *cells, const int *obstacles) {
    int ii, offset;
    double *speeds;
    const double w1 = params.density * params.accel / 9.0;
    const double w2 = params.density * params.accel / 36.0;

    #pragma acc parallel loop private(ii, offset, speeds) present(cells, obstacles)
    for (ii = 0; ii < params.ny; ii++) {
        offset = ii * params.nx;
        speeds = cells[offset].speeds;
        if (!obstacles[offset] && (speeds[3] - w1) > 0.0 && (speeds[6] - w2) > 0.0 && (speeds[7] - w2) > 0.0) {
            speeds[1] += w1;
            speeds[5] += w2;
            speeds[8] += w2;
            speeds[3] -= w1;
            speeds[6] -= w2;
            speeds[7] -= w2;
        }
    }

    return EXIT_SUCCESS;
}

int collision(const t_param params, t_speed *src_cells, t_speed *dst_cells, const int *obstacles) {
    int ii, jj, kk, offset;
    const double w0 = 4.0 / 9.0;
    const double w1 = 1.0 / 9.0;
    const double w2 = 1.0 / 36.0;
    double u_x, u_y;
    double u[NSPEEDS];
    double d_equ[NSPEEDS];
    double u_sq;
    double local_density;
    int x_e, x_w, y_n, y_s;
    double *dst_speeds;
    double speeds[NSPEEDS];

    #pragma acc parallel loop collapse(2) private(ii, jj, kk, offset, speeds, dst_speeds, local_density, u_x, u_y, u_sq, u, d_equ, x_e, x_w, y_n, y_s) present(src_cells, dst_cells, obstacles)
    for (ii = 0; ii < params.ny; ii++) {
        for (jj = 0; jj < params.nx; jj++) {
            offset = ii * params.nx + jj;
            dst_speeds = dst_cells[offset].speeds;

            y_s = (ii + 1) % params.ny;
            x_w = (jj + 1) % params.nx;
            y_n = (ii == 0) ? (ii + params.ny - 1) : (ii - 1);
            x_e = (jj == 0) ? (jj + params.nx - 1) : (jj - 1);

            if (obstacles[ii * params.nx + jj]) {
                dst_speeds[0] = src_cells[ii * params.nx + jj].speeds[0];
                dst_speeds[1] = src_cells[ii * params.nx + x_w].speeds[3];
                dst_speeds[2] = src_cells[y_s * params.nx + jj].speeds[4];
                dst_speeds[3] = src_cells[ii * params.nx + x_e].speeds[1];
                dst_speeds[4] = src_cells[y_n * params.nx + jj].speeds[2];
                dst_speeds[5] = src_cells[y_s * params.nx + x_w].speeds[7];
                dst_speeds[6] = src_cells[y_s * params.nx + x_e].speeds[8];
                dst_speeds[7] = src_cells[y_n * params.nx + x_e].speeds[5];
                dst_speeds[8] = src_cells[y_n * params.nx + x_w].speeds[6];
            } else {
                local_density = 0.0;
                for (kk = 0; kk < NSPEEDS; kk++) {
                    speeds[kk] = src_cells[offset].speeds[kk];
                    local_density += speeds[kk];
                }

                u_x = (speeds[1] + speeds[5] + speeds[8] - (speeds[3] + speeds[6] + speeds[7])) / local_density;
                u_y = (speeds[2] + speeds[5] + speeds[6] - (speeds[4] + speeds[7] + speeds[8])) / local_density;
                u_sq = u_x * u_x + u_y * u_y;

                for (kk = 0; kk < NSPEEDS; kk++) {
                    u[kk] = 0.0;
                    d_equ[kk] = 0.0;
                }

                d_equ[0] = w0 * local_density * (1.0 - 1.5 * u_sq);
                d_equ[1] = w1 * local_density * (1.0 + 3.0 * u_x + 9.0 / 2.0 * u_x * u_x - 1.5 * u_sq);
                d_equ[2] = w1 * local_density * (1.0 + 3.0 * u_y + 9.0 / 2.0 * u_y * u_y - 1.5 * u_sq);
                d_equ[3] = w1 * local_density * (1.0 + 3.0 * u_x + 3.0 * u_y + 9.0 / 2.0 * (u_x + u_y) * (u_x + u_y) - 1.5 * u_sq);
                d_equ[4] = w1 * local_density * (1.0 - 3.0 * u_y + 9.0 / 2.0 * (u_y - u_y) * (u_y - u_y) - 1.5 * u_sq);
                d_equ[5] = w2 * local_density * (1.0 + 3.0 * (u_x + u_y) + 9.0 / 2.0 * (u_x + u_y) * (u_x + u_y) - 1.5 * u_sq);
                d_equ[6] = w2 * local_density * (1.0 + 3.0 * (-u_x + u_y) + 9.0 / 2.0 * (-u_x + u_y) * (-u_x + u_y) - 1.5 * u_sq);
                d_equ[7] = w2 * local_density * (1.0 - 3.0 * (u_x - u_y) + 9.0 / 2.0 * (u_x - u_y) * (u_x - u_y) - 1.5 * u_sq);
                d_equ[8] = w2 * local_density * (1.0 - 3.0 * (-u_x - u_y) + 9.0 / 2.0 * (-u_x - u_y) * (-u_x - u_y) - 1.5 * u_sq);

                for (kk = 0; kk < NSPEEDS; kk++) {
                    dst_speeds[kk] = (1.0 - params.omega) * speeds[kk] + params.omega * d_equ[kk];
                }
            }
        }
    }

    return EXIT_SUCCESS;
}

int write_values(t_param params, t_speed *cells, int *obstacles, double *av_vels) {
    FILE *f;
    int ii, jj;
    int offset;

    f = fopen(finalStateFile, "wb");
    if (f == NULL) {
        die("Cannot open file", __LINE__, __FILE__);
    }

    for (ii = 0; ii < params.ny; ii++) {
        for (jj = 0; jj < params.nx; jj++) {
            offset = ii * params.nx + jj;
            if (obstacles[offset]) {
                fputc('x', f);
            } else {
                fputc('.', f);
            }
        }
        fputc('\n', f);
    }
    fclose(f);

    f = fopen(avVelocityFile, "w");
    if (f == NULL) {
        die("Cannot open file", __LINE__, __FILE__);
    }

    for (ii = 0; ii < params.maxIters; ii++) {
        fprintf(f, "%.10lf\n", av_vels[ii]);
    }
    fclose(f);

    return EXIT_SUCCESS;
}

void die(const char *message, const int line, const char *file) {
    fprintf(stderr, "Error: %s in file %s at line %d\n", message, file, line);
    exit(EXIT_FAILURE);
}
