struct PairParams;
struct PaArray;
struct RNDunif;
struct Coords;
struct Sdf;
struct int3;
struct float4;

struct WallQuants {
    float4 *pp; /* particle positions xyzo xyzo ... */
    int n;      /* number of particles              */
    int3 L;     /* subdomain size                   */
};

struct WallTicket;

void wall_force(const PairParams*, Wvel_v wv, const Coords *c, Sdf *qsdf, const WallQuants *q, const WallTicket *t, int n, const PaArray *parray, Force *ff);

void wall_ini_quants(int3 L, WallQuants *q);
void wall_ini_ticket(int3 L, WallTicket **t);

void wall_fin_quants(WallQuants *q);
void wall_fin_ticket(WallTicket *t);

void wall_gen_quants(MPI_Comm cart, int maxn, const Sdf *qsdf, /**/ int *n, Particle* pp, WallQuants *q);
void wall_strt_quants(const Coords *coords, int maxn, WallQuants *q);

void wall_gen_ticket(const WallQuants *q, WallTicket *t);

void wall_strt_dump_templ(const Coords *coords, const WallQuants *q);
