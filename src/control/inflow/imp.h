struct Inflow {
    curandState_t *rnds; /* random states on device     */
    float3 *flux;        /* target flux                 */
    float *cumflux;      /* cumulative flux             */
    int n;
};

void ini(Inflow *i);
void fin(Inflow *i);

void create_pp(Inflow *i, int *n, Cloud *c);


