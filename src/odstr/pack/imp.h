void pack_pp(const Particle *pp, int n, int **iidx, int *strt, float2 **dev) {
    KL((dev::pack<float2, 3>), (k_cnf(3*n)),((float2*)pp, iidx, strt, /**/ dev));
}
