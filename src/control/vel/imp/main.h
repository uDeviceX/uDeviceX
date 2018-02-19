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

static void reini_sampler(/**/ PidVCont *c) {
    int3 L = c->L;
    int ncells = L.x * L.y * L.z;

    if (ncells) {
        CC(d::MemsetAsync(c->gridvel, 0, ncells * sizeof(float3)));
        CC(d::MemsetAsync(c->gridnum, 0, ncells * sizeof(int)));
    }

    c->nsamples = 0;
}

static void ini(MPI_Comm comm, int3 L, float3 vtarget, float factor, /**/ PidVCont *c) {
    int ncells, nchunks, rank;

    MC(m::Comm_rank(comm, &rank));
    
    c->L = L;
    c->target = vtarget;
    c->current = make_float3(0, 0, 0);
    c->factor = factor;
    c->Kp = 2;
    c->Ki = 1;
    c->Kd = 8;

    MC(m::Comm_dup(comm, &c->comm));

    ncells = L.x * L.y * L.z;
    CC(d::Malloc((void **) &c->gridvel, ncells * sizeof(float3)));
    CC(d::Malloc((void **) &c->gridnum, ncells * sizeof(int)));

    nchunks = ceiln(ncells, WARPSIZE);
    
    CC(d::alloc_pinned((void **) &c->totvel, nchunks * sizeof(float3)));
    CC(d::alloc_pinned((void **) &c->totnum, nchunks * sizeof(int)));
    CC(d::HostGetDevicePointer((void **) &c->dtotvel, c->totvel, 0));
    CC(d::HostGetDevicePointer((void **) &c->dtotnum, c->totnum, 0));

    c->f = c->sume = make_float3(0, 0, 0);
    c->olde = vtarget;
    
    reini_sampler(/**/ c);

    ini_dump(rank, /**/ &c->fdump);

    c->type = TYPE_NONE;
}

void vcont_ini(MPI_Comm comm, int3 L, float3 vtarget, float factor, /**/ PidVCont **c) {
    PidVCont *vc;
    UC(emalloc(sizeof(PidVCont), (void**) c));
    vc = *c;
    UC(ini(comm, L, vtarget, factor, /**/ vc));
}

void vcont_fin(/**/ PidVCont *c) {
    CC(d::Free(c->gridvel));
    CC(d::Free(c->gridnum));
    CC(d::FreeHost(c->totvel));
    CC(d::FreeHost(c->totnum));
    MC(m::Comm_free(&c->comm));
    fin_dump(c->fdump);
    UC(efree(c));
}

void vcon_set_cart(/**/ PidVCont *cont) {
    cont->type = TYPE_CART;
}

void vcon_set_radial(/**/ PidVCont *cont) {
    cont->type = TYPE_RAD;
}

void vcont_sample(const Coords *coords, int n, const Particle *pp, const int *starts, const int *counts, /**/ PidVCont *c) {
    int3 L = c->L;
    Coords_v coordsv;
    dim3 block(8, 8, 1);
    dim3 grid(ceiln(L.x, block.x),
              ceiln(L.y, block.y),
              ceiln(L.z, block.z));

    coords_get_view(coords, &coordsv);
    
    switch (c->type) {
    case TYPE_NONE:
        break;
    case TYPE_CART:
        KL(vcont_dev::sample, (grid, block), (coordsv, c->trans.cart, L, starts, counts, pp, /**/ c->gridvel, c->gridnum));
        break;
    case TYPE_RAD:
        KL(vcont_dev::sample, (grid, block), (coordsv, c->trans.rad, L, starts, counts, pp, /**/ c->gridvel, c->gridnum));
        break;
    default:
        ERR("Unknown type");
        break;
    };
    
    c->nsamples ++;
}

float3 vcont_adjustF(/**/ PidVCont *c) {
    int3 L = c->L;
    int ncells, nchunks, i;
    ncells = L.x * L.y * L.z;
    nchunks = ceiln(ncells, WARPSIZE);

    KL(vcont_dev::reduceByWarp, (nchunks, WARPSIZE), (c->gridvel, c->gridnum, ncells, /**/ c->dtotvel, c->dtotnum));
    dSync();

    float3 e, de;
    double3  vcur = make_double3(0, 0, 0);
    long ncur = 0;

    for (i = 0; i < nchunks; ++i) {
        add(c->totvel + i, /**/ &vcur);
        ncur += c->totnum[i];
    }

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

    reini_sampler(/**/c);

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
