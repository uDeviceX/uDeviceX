struct Outflow {
    int *kk; /* die or stay alive? */
};

void ini(int maxp, /**/ Outflow *o);
void fin(/**/ Outflow *o);

void filter_particles(int n, const Particle *pp, Outflow *o);
