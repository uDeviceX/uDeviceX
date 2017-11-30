void ini(int2 nc, Inflow **i) {
    int n;
    size_t sz;
    Inflow *ip;

    UC(emalloc(sizeof(Inflow), (void**) i));
    
    ip = *i;
    
    n = nc.x * nc.y;
    ip->nc = nc;
    
    sz = n * sizeof(curandState_t);
    CC(d::Malloc((void**) &ip->rnds, sz));

    sz = n * sizeof(float3);
    CC(d::Malloc((void**) &ip->uu, sz));

    sz = n * sizeof(float);
    CC(d::Malloc((void**) &ip->cumflux, sz));

    sz = sizeof(int);
    CC(d::Malloc((void**) &ip->ndev, sz));

    // TODO 
    plate::Params p;
    plate::VParams vp;

    p.o = make_float3(-XS/2, 0,     -ZS/2);
    p.a = make_float3(    0,  YS/2,     0);
    p.b = make_float3(    0,     0,  ZS);

    vp.u = make_float3(10.f, 0, 0);
    vp.upoiseuille = true;
    vp.upoiseuille = false;

    KL(plate::ini_vel, (k_cnf(nc.x * nc.y)), (vp, p, nc, /**/ ip->uu));
}

void fin(Inflow *i) {
    CC(d::Free(i->rnds));
    CC(d::Free(i->uu));
    CC(d::Free(i->cumflux));
    CC(d::Free(i->ndev));
}


void create_pp(Inflow *i, int *n, Particle *pp) {
    int2 nc;

    nc = i->nc;
    
    CC(d::MemcpyAsync(i->ndev, n, sizeof(int), H2D));
    
    // TODO
    plate::Params p;
    p.o = make_float3(-XS/2, 0,     -ZS/2);
    p.a = make_float3(    0,  YS/2,     0);
    p.b = make_float3(    0,     0,  ZS);

    KL(plate::cumulative_flux, (k_cnf(nc.x * nc.y)), (p, nc, i->uu, /**/ i->cumflux));
    KL(plate::create_particles, (k_cnf(nc.x * nc.y)),
       (p, nc, i->uu, /*io*/ i->rnds, i->cumflux, /**/ i->ndev, pp));
    
    CC(d::MemcpyAsync(n, i->ndev, sizeof(int), D2H));
    dSync(); // wait for n
}
