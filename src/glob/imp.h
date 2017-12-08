struct Coords {
    int c[3]; /* rank coordinates */
    int d[3]; /* rank sizes       */
};

void ini(MPI_Comm cart, Coords *c);
void fin(Coords *c);

void local2center(const Coords *c, float3 rl, /**/ float3 *rc);
void center2local(const Coords *c, float3 rc, /**/ float3 *rl);

void local2global(const Coords *c, float3 rl, /**/ float3 *rg);
void global2local(const Coords *c, float3 rg, /**/ float3 *rl);
