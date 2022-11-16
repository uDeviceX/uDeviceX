void scheme_move_apply(float dt, float mass, int n, const Force *ff, Particle *pp) {
    KL(scheme_move_dev::update, (k_cnf(n)), (dt, mass, pp, ff, n));
}

void scheme_move_clear_vel(int n, /**/ Particle *pp) {
    KL(scheme_move_dev::clear_vel, (k_cnf(n)), (n, /**/ pp));
}
