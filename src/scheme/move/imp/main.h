void scheme_move_apply(MoveParams * par, float mass, int n, const Force *ff, Particle *pp) {

    MoveParams_v parv;
    parv = scheme_move_params_get_view(par);

    KL(dev::update, (k_cnf(n)), (parv, mass, pp, ff, n));
}

void scheme_move_clear_vel(int n, /**/ Particle *pp) {
    KL(dev::clear_vel, (k_cnf(n)), (n, /**/ pp));
}
