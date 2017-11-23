void main(float mass, Param fpar, int n, const Particle *pp, /**/ Force* ff) {
    KL(dev::main, (k_cnf(n)), (mass, fpar, n, pp, /**/ ff));
}
