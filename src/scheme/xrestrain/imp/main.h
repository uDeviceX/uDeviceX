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

void scheme_restrain_apply(MPI_Comm, const Restrain*, const int *cc, long it, /**/ SchemeQQ);
