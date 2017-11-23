void force(float mass, Fparams fpar, int n, const Particle *pp, /**/ Force* ff) {
    KL(dev::force, (k_cnf(n)), (mass, fpar, n, pp, /**/ ff));
}
