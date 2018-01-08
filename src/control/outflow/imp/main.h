static void reset_ndead(Outflow *o) {
    CC(d::MemsetAsync(o->ndead_dev, 0, sizeof(int)));
    o->ndead = 0;
}

void ini(int maxp, /**/ Outflow **o0) {
    Outflow *o;
    size_t sz;

    UC(emalloc(sizeof(Outflow), (void**) o0));
    o = *o0;
    
    sz = maxp * sizeof(int);
    CC(d::Malloc((void**) &o->kk, sz));
    CC(d::Malloc((void**) &o->ndead_dev, sizeof(int)));

    CC(d::MemsetAsync(o->kk, 0, sz));
    reset_ndead(o);
    o->type = TYPE_NONE;
}

void fin(/**/ Outflow *o) {
    CC(d::Free(o->kk));
    CC(d::Free(o->ndead_dev));
    UC(efree(o));
}

void filter_particles(int n, const Particle *pp, /**/ Outflow *o) {
    reset_ndead(o);

    switch(o->type) {
    case TYPE_CIRCLE:
        KL(filter, (k_cnf(n)), (n, pp, o->params.circle, /**/ o->ndead_dev, o->kk) );
        break;
    case TYPE_PLANE:
        KL(filter, (k_cnf(n)), (n, pp, o->params.plane, /**/ o->ndead_dev, o->kk) );
        break;
    case TYPE_NONE:
    default:
        ERR("No type set");
        break;
    };
}

void download_ndead(Outflow *o) {
    CC(d::Memcpy(&o->ndead, o->ndead_dev, sizeof(int), D2H));
    // printf("killed %d particles\n", o->ndead);
}

int* get_deathlist(Outflow *o) {return o->kk;}
int  get_ndead(Outflow *o)     {return o->ndead;}
