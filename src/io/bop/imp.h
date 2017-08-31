namespace bop {
struct Ticket {
    float *w_pp;    // particle workspace
    int mi[3];      // global coordinates of my origin
    MPI_Datatype dumptype;
};

void ini(Ticket *t);
void fin(Ticket *t);

void parts(const Particle *pp, const long n, const char *name, const int step, Ticket *t);
void intdata(const int *ii, const long n, const char *name, const int step);
}
