void area_volume_compute(int nt, int nv, int nc, const Particle* pp, const int4 *tri, /**/ float *av) {
    dim3 avThreads(256, 1);
    dim3 avBlocks(1, nc);
    Dzero(av, 2*nc);
    KL(dev::main, (avBlocks, avThreads), (nt, nv, pp, tri, av));
}
