void dev(int nt, int nv, int nc, const Texo<float2> texvert, const int4 *textri, /**/ float *av) {
    dim3 avThreads(256, 1);
    dim3 avBlocks(1, nc);
    Dzero(av, 2*nc);
    KL(dev0::dev, (avBlocks, avThreads), (nt, nv, texvert, textri, av));
}

void hst(int nt, int nv, int nc, const Texo<float2> texvert, const int4 *textri, /**/ float *hst) {
    float *dev;
    Dalloc(&dev, 2*nc);
    area_volume::dev(nt, nv, nc, texvert, textri, /**/ dev);
    cD2H(hst, dev, 2*nc);
    Dfree(dev);
}

