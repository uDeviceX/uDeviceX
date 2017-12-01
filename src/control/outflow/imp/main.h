static void reset_ndead(Outflow *o) {
    CC(d::MemsetAsync(o->ndead_dev, 0, sizeof(int)));
    o->ndead = 0;
}

void ini(int maxp, /**/ Outflow **o0) {
    Outflow *o;
    size_t sz;

    UC(emalloc(sizeof(Outflow), (void**) o0));
    o = *o0;
    
    sz = maxp * sizeof(Particle);
    CC(d::Malloc((void**) &o->kk, sz));
    CC(d::Malloc((void**) &o->ndead_dev, sizeof(int)));

    CC(d::MemsetAsync(o->kk, 0, sz));
    reset_ndead(o);
}

void fin(/**/ Outflow *o) {
    CC(d::Free(o->kk));
    CC(d::Free(o->ndead_dev));
    UC(efree(o));
}


void filter_particles_circle(float R, int n, const Particle *pp, Outflow *o) {
    circle::Params params;
    float3 origin;

    params.Rsq = R*R;
    // TODO
    origin.x = 0;
    origin.y = 0;
    origin.z = 0;

    reset_ndead(o);    
    KL(circle::filter, (k_cnf(n)), (origin, n, pp, params, /**/ o->kk, o->ndead_dev) );
}

void filter_particles_plane(float3 normal, float3 r, int n, const Particle *pp, Outflow *o) {
    plane::Params params;
    float3 origin;

    params.a = normal.x;
    params.b = normal.y;
    params.c = normal.z;
    params.d = - (normal.x * r.x + normal.y * r.y + normal.z * r.z);
    // TODO
    origin.x = 0;
    origin.y = 0;
    origin.z = 0;

    reset_ndead(o);
    KL(plane::filter, (k_cnf(n)), (origin, n, pp, params, /**/ o->kk, o->ndead_dev) );
}

void download_ndead(Outflow *o) {
    CC(d::Memcpy(&o->ndead, o->ndead_dev, sizeof(int), D2H));
}
