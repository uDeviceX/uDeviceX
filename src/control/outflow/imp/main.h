void ini(int maxp, /**/ Outflow *o) {
    size_t sz = maxp * sizeof(Particle);
    CC(d::Malloc((void**) &o->kk, sz));
}

void fin(/**/ Outflow *o) {
    CC(d::Free(o->kk));
}

void filter_particles_circle(float R, int n, const Particle *pp, Outflow *o) {
    circle::Params params;
    float3 origin;

    params.Rsq = R*R;
    // TODO
    origin.x = 0;
    origin.y = 0;
    origin.z = 0;
    
    KL(circle::filter, (k_cnf(n)), (origin, n, pp, params, /**/ o->kk) );
}
