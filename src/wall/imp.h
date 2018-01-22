struct RNDunif;

namespace wall {
struct WallQuants {
    float4 *pp; /* particle positions xyzo xyzo ... */
    int n;      /* number of particles              */
};

struct Ticket {
    RNDunif *rnd;        /* rng on host                                        */
    Clist cells;         /* cell lists (always the same, no need to store map) */
    Texo<int> texstart;  /* texture of starts from clist                       */
    Texo<float4> texpp;  /* texture of particle positions                      */
};

namespace grey {
void force(Wvel_v wv, Coords c, Sdf *qsdf, const WallQuants q, const Ticket t, Cloud cloud, const int n, Force *ff);
}

namespace color {
void force(Wvel_v wv, Coords c, Sdf *qsdf, const WallQuants q, const Ticket t, Cloud cloud, const int n, Force *ff);
}

void alloc_quants(WallQuants *q);
void alloc_ticket(Ticket *t);

void free_quants(WallQuants *q);
void free_ticket(Ticket *t);

void gen_quants(MPI_Comm cart, int maxn, Sdf *qsdf, /**/ int *n, Particle* pp, WallQuants *q);
void strt_quants(Coords coords, int maxn, WallQuants *q);

void gen_ticket(const WallQuants q, Ticket *t);

void strt_dump_templ(Coords coords, const WallQuants q);

}
