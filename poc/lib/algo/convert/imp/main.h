void convert_pp2rr_current (long n, const Particle *pp, Positioncp *rr) {
    KL(convert_dev::pp2rr_current, (k_cnf(n)), (n, pp, rr));
}

void convert_pp2rr_previous(long n, const Particle *pp, Positioncp *rr) {
    KL(convert_dev::pp2rr_previous, (k_cnf(n)), (n, pp, rr));
}

void convert_pp2f4_pos(int n, const float *pp, /**/ float4 *zpp) {
    static_assert(sizeof(Particle) == 6 * sizeof(float),
                  "sizeof(Particle) != 6 * sizeof(float)");
    KL(convert_dev::pp2f4_pos, (k_cnf(n)), (n, pp, zpp));
}
