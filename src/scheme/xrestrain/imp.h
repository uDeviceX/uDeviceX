struct Particle;
struct Config;
struct Restrain;

/* quantities and sizes wrapper */
struct SchemeQQ {
    Particle *o, *r;
    int on, rn;
};

void scheme_restrain_ini(Restrain**);
void scheme_restrain_fin(Restrain*);

void scheme_restrain_set_red(Restrain*);
void scheme_restrain_set_rbc(Restrain*);
void scheme_restrain_set_none(Restrain*);
void scheme_restrain_set_freq(int freq, Restrain*);

void scheme_restrain_set_conf(const Config*, Restrain*);

void scheme_restrain_apply(MPI_Comm comm, const int *cc, long it, /**/ Restrain *r, SchemeQQ qq);

