void main(int nt, int nv, int nc, const Texo<float2> texvert, const Texo<int4> textri, /**/ float *av) {
    dim3 avThreads(256, 1);
    dim3 avBlocks(1, nc);
    Dzero(av, 2*nc);
    KL(dev::area_volume, (avBlocks, avThreads), (nt, nv, texvert, textri, av));
}

void main_hst(int nt, int nv, int nc, const Texo<float2> texvert, const Texo<int4> textri, /**/ float *hst) {
    float *dev;
    Dalloc(&dev, 2*nc);
    area_volume::main(nt, nv, nc, texvert, textri, /**/ dev);
    cD2H(hst, dev, 2*nc);
    Dfree(dev);
}

