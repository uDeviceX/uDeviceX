namespace sub {
void ini(MPI_Comm, cudaArray*, tex3Dca*);
void bulk_wall(const tex3Dca, /*io*/ Particle *s_pp, int *s_n, /*o*/ Particle *w_pp, int *w_n);
int  who_stays(const tex3Dca, Particle*, int n, int nc, int nv, /**/ int *stay);
}
