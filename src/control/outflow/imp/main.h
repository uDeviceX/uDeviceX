static void reset_ndead(Outflow *o) {
    CC(d::MemsetAsync(o->ndead_dev, 0, sizeof(int)));
    o->ndead = 0;
}

void outflow_ini(int maxp, /**/ Outflow **o0) {
    Outflow *o;
    size_t sz;

    EMALLOC(1, o0);
    o = *o0;
    
    sz = maxp * sizeof(int);
    CC(d::Malloc((void**) &o->kk, sz));
    CC(d::Malloc((void**) &o->ndead_dev, sizeof(int)));

    CC(d::MemsetAsync(o->kk, 0, sz));
    reset_ndead(o);
    o->type = TYPE_NONE;
}

void outflow_fin(/**/ Outflow *o) {
    CC(d::Free(o->kk));
    CC(d::Free(o->ndead_dev));
    EFREE(o);
}

void outflow_ini_params_circle(const Coords *coords, float3 c, float R, Outflow *o) {
    enum {X, Y, Z};
    ParamsCircle p;
    p.Rsq = R*R;
    
    // shift to local referential
    global2local(coords, c, /**/ &p.c);

    o->params.circle = p;
    o->type = TYPE_CIRCLE;
}

void outflow_ini_params_plate(const Coords *c, int dir, float r0, Outflow *o) {
    enum {X, Y, Z};
    ParamsPlate p = {0, 0, 0, 0};    

    switch (dir) {
    case X:
        p.a = 1;
        p.d = xg2xl(c, r0);
        break;
    case Y:
        p.b = 1;
        p.d = yg2yl(c, r0);
        break;
    case Z:
        p.c = 1;
        p.d = zg2zl(c, r0);
        break;
    default:
        ERR("undefined direction %d\n", dir);
        break;
    };    
    
    o->params.plate = p;
    o->type = TYPE_PLATE;
}

void outflow_filter_particles(int n, const Particle *pp, /**/ Outflow *o) {
    reset_ndead(o);

    switch(o->type) {
    case TYPE_CIRCLE:
        KL(filter, (k_cnf(n)), (n, pp, o->params.circle, /**/ o->ndead_dev, o->kk) );
        break;
    case TYPE_PLATE:
        KL(filter, (k_cnf(n)), (n, pp, o->params.plate, /**/ o->ndead_dev, o->kk) );
        break;
    case TYPE_NONE:
    default:
        ERR("No type set");
        break;
    };
}

void outflow_download_ndead(Outflow *o) {
    CC(d::Memcpy(&o->ndead, o->ndead_dev, sizeof(int), D2H));
    // printf("killed %d particles\n", o->ndead);
}

int* outflow_get_deathlist(Outflow *o) {return o->kk;}
int  outflow_get_ndead(Outflow *o)     {return o->ndead;}
