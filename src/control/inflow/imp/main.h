void ini(int2 nc, Inflow *i) {
    int n;
    size_t sz;
    n = nc.x * nc.y;
    sz = n * sizeof(curandState_t);

    CC(d::Malloc((void**) &i->rnds, sz));

    sz = n * sizeof(float3);
    CC(d::Malloc((void**) &i->flux, sz));

    sz = n * sizeof(float);
    CC(d::Malloc((void**) &i->cumflux, sz));

    sz = sizeof(int);
    CC(d::Malloc((void**) &i->ndev, sz));
}

void fin(Inflow *i) {
    CC(d::Free(i->rnds));
    CC(d::Free(i->flux));
    CC(d::Free(i->cumflux));
    CC(d::Free(i->ndev));
}

void create_pp(Inflow *i, int *n, Cloud *c) {
    
}
