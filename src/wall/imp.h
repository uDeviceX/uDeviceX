struct PairParams;
struct PaArray;
struct FoArray;
struct WvelStep;
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

void wall_force(const PairParams*, const WvelStep *, const Coords*,
                Sdf*, const WallQuants*, const WallTicket*, int n, const PaArray*, const FoArray*);

void wall_ini_quants(int3 L, WallQuants*);
void wall_ini_ticket(int3 L, WallTicket**);

void wall_fin_quants(WallQuants*);
void wall_fin_ticket(WallTicket*);

void wall_gen_quants(MPI_Comm, int maxn, const Sdf *qsdf, /**/ int *n, Particle*, WallQuants*);
void wall_strt_quants(const Coords*, int maxn, WallQuants*);

void wall_gen_ticket(const WallQuants*, WallTicket*);

void wall_strt_dump_templ(const Coords*, const WallQuants*);
