void ini(int maxp, /**/ Outflow *o) {
    size_t sz = maxp * sizeof(Particle);
    CC(d::Malloc((void**) &o->kk, sz));
}

void fin(/**/ Outflow *o) {
    CC(d::Free(o->kk));
}

void filter_particles(int n, const Particle *pp, Outflow *o) {
    circle::Params params;
    float3 origin;

    params.Rsq = 4; // TODO
    params.inside = 1;
    origin.x = 0;
    origin.y = 0;
    origin.z = 0;
    
    KL(circle::filter, (k_cnf(n)), (origin, n, pp, params, /**/ o->kk) );
}
