void main(float mass, Param par, int n, const Particle *pp, /**/ Force* ff) {
    float alpha;
    alpha = par.a;
    KL(dev::main, (k_cnf(n)), (mass, alpha, n, pp, /**/ ff));
}
