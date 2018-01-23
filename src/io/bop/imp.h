namespace bop {
struct BopWork {
    float *w_pp;    // particle workspace
};

void ini(MPI_Comm comm, BopWork *t);
void fin(BopWork *t);

void parts(MPI_Comm cart, const Coords *coords, const Particle *pp, long n, const char *name, int step, /*w*/ BopWork *t);
void parts_forces(MPI_Comm cart, const Coords *coords, const Particle *pp, const Force *ff, long n, const char *name, int step, /*w*/ BopWork *t);
void ids(MPI_Comm cart, const int *ii, long n, const char *name, int step);
void colors(MPI_Comm cart, const int *ii, long n, const char *name, int step);
}
