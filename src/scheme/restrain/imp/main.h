void scheme_restrain_ini(Restrain **rstr) {
    Restrain *r;
    EMALLOC(1, rstr);
    r = *rstr;

    CC(d::alloc_pinned((void**) &r->n, sizeof(int)));
    CC(d::alloc_pinned((void**) &r->v, sizeof(float3)));
}

void scheme_restrain_fin(Restrain *r) {
    CC(d::FreeHost(r->n));
    CC(d::FreeHost(r->v));
    EFREE(r);
}

void scheme_restrain_set_red(Restrain *r) {
    r->kind = RSTR_COL;
}

void scheme_restrain_set_rbc(Restrain *r) {
    r->kind = RSTR_RBC;
}

void scheme_restrain_set_none(Restrain *r) {
    r->kind = RSTR_NONE;
}

void scheme_restrain_set_freq(int freq, Restrain *r) {
    r->freq = freq;
}

static const char* kind2tag(int i) {
    static const char *tags[] = {"NONE", "RED", "RBC"};
    return tags[i];
}

static void print_stats(const char *tag, int n, float3 v) {
    msg_print("restrain %s: n = %d [% .3e % .3e % .3e]", tag, n, v.x, v.y, v.z);
}

static void report(const Restrain *r, long it) {
    int n, freq, cond, kind;
    float3 v;
    kind = r->kind;
    freq = r->freq;
    cond = freq > 0 && it % freq == 0 && kind != RSTR_NONE;
    n = *(r->n);
    v = *(r->v);
    if (cond)
        print_stats(kind2tag(kind), n, v);
}

static void reini(Restrain *r) {
    *(r->n) = 0;
    *(r->v) = make_float3(0, 0, 0);
}

static int all_reduce(MPI_Comm comm, int n) {
    int ntot = 0;
    MC(m::Allreduce(&n, &ntot, 1, MPI_INT, MPI_SUM, comm));
    return ntot;
}

static float3 all_reduce(MPI_Comm comm, float3 v) {
    float3 vtot = make_float3(0, 0, 0);
    MC(m::Allreduce(&v.x, &vtot.x, 3, MPI_FLOAT, MPI_SUM, comm));
    return vtot;
}

template <typename Map>
static void apply(MPI_Comm comm, Map m, int np, /**/ Particle *pp, Restrain *r) {
    int n;
    float3 v;
    float scale;

    reini( /**/ r);
    KL(restrain_dev::sum, (k_cnf(np)), (m, np, pp, /**/ r->n, r->v));
    dSync(); /* wait for n and v to be computed */

    n = *(r->n);
    v = *(r->v);    

    n = all_reduce(comm, n);
    v = all_reduce(comm, v);

    scale = n ? 1.0 / n : 1;
    v.x *= scale;
    v.y *= scale;
    v.z *= scale;    
    
    KL(restrain_dev::shift, (k_cnf(np)), (m, v, np, /**/ pp));

    *(r->v) = v;
    *(r->n) = n;
}

void scheme_restrain_apply(MPI_Comm comm, const int *cc, long it, /**/ Restrain *r, SchemeQQ qq) {
    int kind;
    kind = r->kind;
    switch (kind) {
    case RSTR_NONE:
        break;
    case RSTR_COL:
        restrain_dev::MapColor mc;
        mc.cc = cc;
        mc.color = RED_COLOR;
        apply(comm, mc, qq.on, /**/ qq.o, r);
        report(r, it);
        break;
    case RSTR_RBC:
        restrain_dev::MapGrey mg;
        apply(comm, mg, qq.rn, /**/ qq.r, r);
        report(r, it);
        break;
    default:
        ERR("Unknown kind <%d>", kind);
    };
}
