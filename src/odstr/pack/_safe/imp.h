void pack_pp(const Particle *pp, int n, int **iidx, int *strt, float2 **dev) {
    dbg::check_pos_pu(pp, n, __FILE__, __LINE__, "pack.safe: before");
    KL((dev::pack<float2, 3>), (k_cnf(3*n)),((float2*)pp, iidx, strt, /**/ dev));
    dbg::check_pos_pu(pp, n, __FILE__, __LINE__, "after.safe: after");
}
