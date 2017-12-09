void main(Coords c, float mass, Param par, int n, const Particle *pp, /**/ Force* ff) {
    float ax, ay, az;
    ax = par.a; ay = par.b; az = par.c;
    KL(dev::main, (k_cnf(n)), (mass, ax, ay, az, n, pp, /**/ ff));
}
