struct RNDunif;
struct Coords;

struct WallQuants {
    float4 *pp; /* particle positions xyzo xyzo ... */
    int n;      /* number of particles              */
};

struct WallTicket;

namespace grey {
void wall_force(Wvel_v wv, const Coords *c, Sdf *qsdf, const WallQuants *q, const WallTicket *t, int n, Cloud cloud, Force *ff);
}

namespace color {
void wall_force(Wvel_v wv, const Coords *c, Sdf *qsdf, const WallQuants *q, const WallTicket *t, int n, Cloud cloud, Force *ff);
}

void wall_ini_quants(WallQuants *q);
void wall_ini_ticket(WallTicket **t);

void wall_fin_quants(WallQuants *q);
void wall_fin_ticket(WallTicket *t);

void wall_gen_quants(MPI_Comm cart, int maxn, Sdf *qsdf, /**/ int *n, Particle* pp, WallQuants *q);
void wall_strt_quants(Coords coords, int maxn, WallQuants *q);

void wall_gen_ticket(const WallQuants *q, WallTicket *t);

void wall_strt_dump_templ(Coords coords, const WallQuants *q);
