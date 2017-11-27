/* return statistics of restrain: number of particle and center of
   mass velocity */
void stats(/**/ int *n, float *v);

namespace color {
void vel(MPI_Comm comm, const int *cc, int color, int n, /**/ Particle *pp);
}

namespace grey {
void vel(MPI_Comm comm, int n, /**/ Particle *pp);
}

