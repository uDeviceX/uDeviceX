void main(float mass, int n, const Force *ff, Particle *pp) {
    KL(dev::update, (k_cnf(n)), (mass, pp, ff, n));
}

void clear_vel(int n, /**/ Particle *pp) {
    KL(dev::clear_vel, (k_cnf(n)), (n, /**/ pp));
}
