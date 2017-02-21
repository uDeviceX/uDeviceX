#include <mpi.h>
#include <sys/resource.h>
#include <utility>
#include <cell-lists.h>
#include <cuda-dpd.h>
#include <cstdio>
#include ".conf.h" /* configuration file (copy from .conf.test.h) */
#include "common.h"

bool Particle::initialized = false;
MPI_Datatype Particle::mytype;
bool Acceleration::initialized = false;
MPI_Datatype Acceleration::mytype;

void CellLists::build(Particle * const p, const int n, cudaStream_t stream, int * const order, const Particle * const src)
{
    if (n > 0)
      build_clists_vanilla((float * )p, n, 1, LX, LY, LZ, -LX/2, -LY/2, -LZ/2, order, start, count,  NULL, stream, (float *)src);
    else
    {
        CC(cudaMemsetAsync(start, 0, sizeof(int) * ncells, stream));
        CC(cudaMemsetAsync(count, 0, sizeof(int) * ncells, stream));
    }
}


void diagnostics(MPI_Comm comm, MPI_Comm cartcomm, Particle * particles, int n, float dt, int idstep) {
    double p[] = {0, 0, 0};
    for(int i = 0; i < n; ++i)
        for(int c = 0; c < 3; ++c)
            p[c] += particles[i].u[c];

    int rank;
    MC(MPI_Comm_rank(comm, &rank) );

    int dims[3], periods[3], coords[3];
    MC(MPI_Cart_get(cartcomm, 3, dims, periods, coords) );

    MC(MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &p, rank == 0 ? &p : NULL, 3, MPI_DOUBLE, MPI_SUM, 0, comm) );

    double ke = 0;
    for(int i = 0; i < n; ++i)
        ke += pow(particles[i].u[0], 2) + pow(particles[i].u[1], 2) + pow(particles[i].u[2], 2);

    MC( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &ke, &ke, 1, MPI_DOUBLE, MPI_SUM, 0, comm) );
    MC( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &n, &n, 1, MPI_INT, MPI_SUM, 0, comm) );

    double kbt = 0.5 * ke / (n * 3. / 2);

    if (rank == 0)
    {
        static bool firsttime = true;
        FILE * f = fopen("diag.txt", firsttime ? "w" : "a");
        firsttime = false;

        if (idstep == 0)
            fprintf(f, "# TSTEP\tKBT\tPX\tPY\tPZ\n");

        printf("\x1b[91m timestep: %e\t%.10e\t%.10e\t%.10e\t%.10e\x1b[0m\n", idstep * dt, kbt, p[0], p[1], p[2]);
        fprintf(f, "%e\t%.10e\t%.10e\t%.10e\t%.10e\n", idstep * dt, kbt, p[0], p[1], p[2]);

        fclose(f);
    }
}
