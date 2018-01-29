struct Particle;
struct Restrain;

/* quantities and sizes wrapper */
struct SchemeQQ {
    Particle *o, *r;
    int on, rn;
};


void scheme_restrain_ini(Restrain**);
void scheme_restrain_fin(Restrain*);

void scheme_restrain_apply(MPI_Comm, const Restrain*, const int *cc, long it, /**/ SchemeQQ);
