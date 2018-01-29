struct Particle;
struct Config;
struct Restrain;

// tag::wrap[]
struct SchemeQQ {
    Particle *o, *r;
    int on, rn;
};
// end::wrap[]

// tag::mem[]
void scheme_restrain_ini(Restrain**);
void scheme_restrain_fin(Restrain*);
// end::mem[]

// tag::set[]
void scheme_restrain_set_red(Restrain*); // <1>
void scheme_restrain_set_rbc(Restrain*); // <2>
void scheme_restrain_set_none(Restrain*); // <3>
void scheme_restrain_set_freq(int freq, Restrain*); // <4>
// end::set[]

// tag::cnf[]
void scheme_restrain_set_conf(const Config*, Restrain*);
// end::cnf[]

// tag::int[]
void scheme_restrain_apply(MPI_Comm comm, const int *cc, long it, /**/ Restrain *r, SchemeQQ qq);
// end::int[]
