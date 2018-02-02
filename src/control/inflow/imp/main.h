static void ini_rnd(int n, curandState_t *rr) {
    long seed = 1234567;
    KL(dev::ini_rnd, (k_cnf(n)), (seed, n, rr));
}

static void ini_flux(int n, curandState_t *rr, float *cumflux) {
    KL(dev::ini_flux, (k_cnf(n)), (n, rr, cumflux));
}

void inflow_ini(int2 nc, Inflow **i) {
    int n;
    size_t sz;
    Inflow *ip;
    Desc *d;

    UC(emalloc(sizeof(Inflow), (void**) i));

    ip = *i;
    d = &ip->d;

    n = nc.x * nc.y;
    d->nc = nc;

    sz = n * sizeof(curandState_t);
    CC(d::Malloc((void**) &d->rnds, sz));

    sz = n * sizeof(float3);
    CC(d::Malloc((void**) &d->uu, sz));

    sz = n * sizeof(float);
    CC(d::Malloc((void**) &d->cumflux, sz));

    sz = sizeof(int);
    CC(d::Malloc((void**) &d->ndev, sz));

    ini_rnd(n, d->rnds);
    ini_flux(n, d->rnds, d->cumflux);

    ip->t = TYPE_NONE;
}

static void ini_velocity(Type t, int2 nc, const ParamsU *p, const VParamsU *vp, /**/ float3 *uu) {
    int n = nc.x * nc.y;
    switch(t) {
    case TYPE_PLATE:
        KL(dev::ini_vel, (k_cnf(n)), (vp->plate, p->plate, nc, /**/ uu));
        break;
    case TYPE_CIRCLE:
        KL(dev::ini_vel, (k_cnf(n)), (vp->circle, p->circle, nc, /**/ uu));
        break;
    case TYPE_NONE:
        break;
    default:
        ERR("No inflow type is set");
        break;
    };
}

void inflow_ini_velocity(Inflow *i) {
    ini_velocity(i->t, i->d.nc, &i->p, &i->vp, /**/ i->d.uu);
}

void inflow_fin(Inflow *i) {
    Desc *d = &i->d;
    CC(d::Free(d->rnds));
    CC(d::Free(d->uu));
    CC(d::Free(d->cumflux));
    CC(d::Free(d->ndev));
    UC(efree(i));
}

static void create_solvent(float dt0, Inflow *i, int *n, SolventWrap wrap) {
    int2 nc;
    Desc *d;
    int nctot;

    d = &i->d;
    nc = d->nc;
    nctot = nc.x * nc.y;

    CC(d::MemcpyAsync(d->ndev, n, sizeof(int), H2D));

    switch(i->t) {
    case TYPE_PLATE:
        KL(dev::cumulative_flux, (k_cnf(nctot)), (dt0, i->p.plate, nc, d->uu, /**/ d->cumflux));
        KL(dev::create_particles, (k_cnf(nctot)),
           (i->p.plate, nc, d->uu, /*io*/ d->rnds, d->cumflux, /**/ d->ndev, wrap));
        break;
    case TYPE_CIRCLE:
        KL(dev::cumulative_flux, (k_cnf(nctot)), (dt0, i->p.circle, nc, d->uu, /**/ d->cumflux));
        KL(dev::create_particles, (k_cnf(nctot)),
           (i->p.circle, nc, d->uu, /*io*/ d->rnds, d->cumflux, /**/ d->ndev, wrap));
        break;
    case TYPE_NONE:
        break;
    default:
        ERR("No inflow type is set");
        break;
    };

    CC(d::MemcpyAsync(n, d->ndev, sizeof(int), D2H));
    dSync(); // wait for n
}

void inflow_create_pp(float dt0, Inflow *i, int *n, Particle *pp) {
    SolventWrap wrap;

    wrap.pp = pp;
    wrap.cc = NULL;
    wrap.multisolvent = false;
    wrap.color = -1;

    create_solvent(dt0, i, n, wrap);
}

void inflow_create_pp_cc(float dt0, int newcolor, Inflow *i, int *n, Particle *pp, int *cc) {
    SolventWrap wrap;

    wrap.pp = pp;
    wrap.cc = cc;
    wrap.multisolvent = true;
    wrap.color = newcolor;
    create_solvent(dt0, i, n, wrap);
}
