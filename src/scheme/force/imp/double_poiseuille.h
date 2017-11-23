void main(float mass, Param par, int n, const Particle *pp, /**/ Force* ff) {
    float f0;
    f0 = par.a;
    KL(dev::main, (k_cnf(n)), (mass, f0, n, pp, /**/ ff));
}
