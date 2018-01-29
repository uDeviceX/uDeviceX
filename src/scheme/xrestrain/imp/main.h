void scheme_restrain_ini(Restrain **rstr) {
    Restrain *r;
    UC(emalloc(sizeof(Restrain), (void**) rstr));
    r = *rstr;

    CC(d::alloc_pinned((void**) &r->n, sizeof(int)));
    CC(d::alloc_pinned((void**) &r->v, sizeof(float3)));
}

void scheme_restrain_fin(Restrain *r) {
    CC(d::FreeHost(r->n));
    CC(d::FreeHost(r->v));
    UC(efree(r));
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

void scheme_restrain_apply(MPI_Comm, const Restrain*, const int *cc, long it, /**/ SchemeQQ) {

}
