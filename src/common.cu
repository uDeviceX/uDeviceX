#include <mpi.h>
#include "l/m.h"

#include <stdio.h>
#include <conf.h>
#include "m.h"     /* MPI */
#include "common.h"

bool Particle::initialized = false;
MPI_Datatype Particle::mytype;

bool Solid::initialized = false;
MPI_Datatype Solid::mytype;

void diagnostics(Particle *pp, int n, int idstep) {
    double p[] = {0, 0, 0};
    for(int i = 0; i < n; ++i)
    for(int c = 0; c < 3; ++c)
    p[c] += pp[i].v[c];

    MC(l::m::Reduce(m::rank == 0 ? MPI_IN_PLACE : &p,
                  m::rank == 0 ? &p : NULL, 3,
                  MPI_DOUBLE, MPI_SUM, 0, m::cart) );
    double ke = 0;
    for(int i = 0; i < n; ++i)
    ke += pow(pp[i].v[0], 2) + pow(pp[i].v[1], 2) + pow(pp[i].v[2], 2);

    MC(l::m::Reduce(m::rank == 0 ? MPI_IN_PLACE : &ke,
                  &ke,
                  1, MPI_DOUBLE, MPI_SUM, 0, m::cart));
    MC(l::m::Reduce(m::rank == 0 ? MPI_IN_PLACE : &n,
                  &n, 1, MPI_INT, MPI_SUM, 0, m::cart));

    double kbt = 0.5 * ke / (n * 3. / 2);
    if (m::rank == 0) {
        static bool firsttime = true;
        FILE * f = fopen("diag.txt", firsttime ? "w" : "a");
        firsttime = false;
        if (idstep == 0) fprintf(f, "# TSTEP\tKBT\tPX\tPY\tPZ\n");
        fprintf(stderr, "%e\t%.10e\t%.10e\t%.10e\t%.10e\n", idstep * dt, kbt, p[0], p[1], p[2]);
        fprintf(f, "%e\t%.10e\t%.10e\t%.10e\t%.10e\n", idstep * dt, kbt, p[0], p[1], p[2]);
        fclose(f);
    }
}
