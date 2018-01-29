/* quantities and sizes wrapper */
struct SchemeQQ {
    Particle *o, *r;
    int on, rn;
};
void scheme_restrain_apply(MPI_Comm comm, const int *cc, long it, /**/ SchemeQQ);
