struct Inflow {
    curandState_t *rnds; /* random states on device         */
    float3 *flux;        /* target flux                     */
    float *cumflux;      /* cumulative flux                 */
    int *ndev;           /* number of particles on device   */
    int2 nc;             /* number of cells in 2 directions */
};

void ini(Inflow *i);
void fin(Inflow *i);

void create_pp(Inflow *i, int *n, Particle *pp);


