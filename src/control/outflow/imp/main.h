void ini(int maxp, /**/ Outflow *o) {
    Dalloc(maxp, /**/ &o->kk);
}

void fin(/**/ Outflow *o) {
    Dfree(o->kk);
}

void filter_particles(int n, const Particle *pp, Outflow *o) {
    // TODO
}
