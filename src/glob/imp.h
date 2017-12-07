struct Coords {
    int c[3]; /* rank coordinates */
    int d[3]; /* rank sizes       */
};

void ini(MPI_Comm cart, Coords *c);
void fin(Coords *c);

void domain_center(const Coords *c, /**/ float3 *rc);
void local2global(const Coords *c, float3 rl, /**/ float3 *rg);
void global2local(const Coords *c, float3 rg, /**/ float3 *rl);
