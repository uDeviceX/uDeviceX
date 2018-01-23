namespace bop {
struct Ticket {
    float *w_pp;    // particle workspace
};

void ini(MPI_Comm comm, Ticket *t);
void fin(Ticket *t);

void parts(MPI_Comm cart, const Coords *coords, const Particle *pp, long n, const char *name, int step, /*w*/ Ticket *t);
void parts_forces(MPI_Comm cart, const Coords *coords, const Particle *pp, const Force *ff, long n, const char *name, int step, /*w*/ Ticket *t);
void ids(MPI_Comm cart, const int *ii, long n, const char *name, int step);
void colors(MPI_Comm cart, const int *ii, long n, const char *name, int step);
}
