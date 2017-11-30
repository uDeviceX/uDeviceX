void ini(int maxp, /**/ Outflow *o) {
    size_t sz = maxp * sizeof(Particle);
    CC(d::Malloc((void**) &o->kk, sz));
    CC(d::Malloc((void**) &o->ndead_dev, sizeof(int)));
}

void fin(/**/ Outflow *o) {
    CC(d::Free(o->kk));
    CC(d::Free(o->ndead_dev));
}

void filter_particles_circle(float R, int n, const Particle *pp, Outflow *o) {
    circle::Params params;
    float3 origin;

    params.Rsq = R*R;
    // TODO
    origin.x = 0;
    origin.y = 0;
    origin.z = 0;
    
    KL(circle::filter, (k_cnf(n)), (origin, n, pp, params, /**/ *o) );
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
    
    KL(plane::filter, (k_cnf(n)), (origin, n, pp, params, /**/ *o) );
}
