enum {WARPSIZE = 32};

static void ini_dump(int rank, /**/ FILE **f) {
    *f = NULL;
    if (rank) return;
    UC(efopen(DUMP_BASE "/vcont.txt", "w", /**/ f));
    fprintf(*f, "#vx vy vz fx fy fz\n");
    msg_print("Velocity controller: dump to " DUMP_BASE "/vcont.txt");
}

static void fin_dump(FILE *f) {
    if (f) UC(efclose(f));
}

static void reini_sampler(/**/ Sampler *s) {
    int3 L = s->L;
    int ncells = L.x * L.y * L.z;

    if (ncells) {
        CC(d::MemsetAsync(s->gridvel, 0, ncells * sizeof(float3)));
        CC(d::MemsetAsync(s->gridnum, 0, ncells * sizeof(int)));
    }

    s->nsamples = 0;
}

static void ini_sampler(int3 L, Sampler *s) {
    int ncells, nchunks;
    s->L = L;
    ncells = L.x * L.y * L.z;

    CC(d::Malloc((void **) &s->gridvel, ncells * sizeof(float3)));
    CC(d::Malloc((void **) &s->gridnum, ncells * sizeof(int)));

    nchunks = ceiln(ncells, WARPSIZE);
    
    CC(d::alloc_pinned((void **) &s->totvel, nchunks * sizeof(float3)));
    CC(d::alloc_pinned((void **) &s->totnum, nchunks * sizeof(int)));
    CC(d::HostGetDevicePointer((void **) &s->dtotvel, s->totvel, 0));
    CC(d::HostGetDevicePointer((void **) &s->dtotnum, s->totnum, 0));
}

static void ini(MPI_Comm comm, int3 L, /**/ PidVCont *c) {
    int rank;

    UC(ini_sampler(L, &c->sampler));
    UC(reini_sampler(&c->sampler));
    
    MC(m::Comm_rank(comm, &rank));
    MC(m::Comm_dup(comm, &c->comm));

    c->current = make_float3(0, 0, 0);
    c->f = c->sume = make_float3(0, 0, 0);
    
    UC(ini_dump(rank, /**/ &c->fdump));

    c->type = TYPE_NONE;
}

void vcont_ini(MPI_Comm comm, int3 L, /**/ PidVCont **c) {
    PidVCont *vc;
    EMALLOC(1, c);
    vc = *c;
    UC(ini(comm, L, /**/ vc));
}

static void fin_sampler(Sampler *s) {
    CC(d::Free(s->gridvel));
    CC(d::Free(s->gridnum));
    CC(d::FreeHost(s->totvel));
    CC(d::FreeHost(s->totnum));
}

void vcont_fin(/**/ PidVCont *c) {
    UC(fin_sampler(&c->sampler));
    MC(m::Comm_free(&c->comm));
    UC(fin_dump(c->fdump));
    EFREE(c);
}

void vcont_set_params(float factor, float Kp, float Ki, float Kd, /**/ PidVCont *c) {
    c->factor = factor;
    c->Kp = Kp;
    c->Ki = Ki;
    c->Kd = Kd;
}

void vcont_set_target(float3 vtarget, /**/ PidVCont *c) {
    c->target = vtarget;
    c->olde   = vtarget;
}

void vcont_set_cart(/**/ PidVCont *cont) {
    cont->type = TYPE_CART;
}

void vcont_set_radial(/**/ PidVCont *cont) {
    cont->type = TYPE_RAD;
}

template <typename Trans>
static void sample(Trans t, const Coords *coords, const Particle *pp, const int *starts, const int *counts, /**/ Sampler *s) {
    int3 L = s->L;
    Coords_v coordsv;

    coords_get_view(coords, &coordsv);

    dim3 block(8, 8, 1);
    dim3 grid(ceiln(L.x, block.x),
              ceiln(L.y, block.y),
              ceiln(L.z, block.z));

    KL(vcont_dev::sample, (grid, block), (coordsv, t, L, starts, counts, pp, /**/ s->gridvel, s->gridnum));
    s->nsamples++;
}

void vcont_sample(const Coords *coords, int n, const Particle *pp, const int *starts, const int *counts, /**/ PidVCont *c) {
    switch (c->type) {
    case TYPE_NONE:
        break;
    case TYPE_CART:
        sample(c->trans.cart, coords, pp, starts, counts, /**/ &c->sampler);
        break;
    case TYPE_RAD:
        sample(c->trans.cart, coords, pp, starts, counts, /**/ &c->sampler);
        break;
    default:
        ERR("Unknown type");
        break;
    };
}

static void reduce_local(Sampler *s, double3 *v, long *n) {
    int3 L = s->L;
    int ncells, nchunks, i;
    ncells = L.x * L.y * L.z;
    nchunks = ceiln(ncells, WARPSIZE);

    KL(vcont_dev::reduceByWarp, (nchunks, WARPSIZE), (s->gridvel, s->gridnum, ncells, /**/ s->dtotvel, s->dtotnum));
    dSync();

    *v = make_double3(0, 0, 0);
    *n = 0;

    for (i = 0; i < nchunks; ++i) {
        add(s->totvel + i, /**/ v);
        *n += s->totnum[i];
    }
}

float3 vcont_adjustF(/**/ PidVCont *c) {
    float3 e, de;
    long ncur;
    double3 vcur;

    UC(reduce_local(&c->sampler, &vcur, &ncur));

    MC(m::Allreduce(MPI_IN_PLACE, &vcur.x, 3, MPI_DOUBLE, MPI_SUM, c->comm));
    MC(m::Allreduce(MPI_IN_PLACE, &ncur,   1, MPI_LONG,   MPI_SUM, c->comm));

    const double fac = ncur ? (1.0 / ncur) : 1.0;
    
    scal(fac, /**/ &vcur);

    c->current = make_float3(vcur.x, vcur.y, vcur.z);

    diff(&c->target, &vcur, /**/ &e);
    diff(&e, &c->olde, /**/ &de);
    add(&e, /**/ &c->sume);

    c->f = make_float3(0, 0, 0);

    axpy(c->factor * c->Kp, &e,       /**/ &c->f);
    axpy(c->factor * c->Ki, &c->sume, /**/ &c->f);
    axpy(c->factor * c->Kd, &de,      /**/ &c->f);

    UC(reini_sampler(&c->sampler));

    c->olde = e;
    return c->f;
}

void vcont_log(const PidVCont *c) {
    if (c->fdump == NULL) return;
    float3 v = c->current;
    float3 f = c->f;
    fprintf(c->fdump, "%.3e %.3e %.3e %.3e %.3e %.3e\n",
            v.x, v.y, v.z, f.x, f.y, f.z);
    fflush(c->fdump);
}
