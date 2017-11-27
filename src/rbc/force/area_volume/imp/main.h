void dev(int nt, int nv, int nc, const Particle* pp, const int4 *tri, /**/ float *av) {
    dim3 avThreads(256, 1);
    dim3 avBlocks(1, nc);
    Dzero(av, 2*nc);
    KL(dev0::dev, (avBlocks, avThreads), (nt, nv, pp, tri, av));
}

void hst(int nt, int nv, int nc, const Particle* pp, const int4 *tri, /**/ float *hst) {
    float *dev;
    Dalloc(&dev, 2*nc);
    area_volume::dev(nt, nv, nc, pp, tri, /**/ dev);
    cD2H(hst, dev, 2*nc);
    Dfree(dev);
}

