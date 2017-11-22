void area_volume(int nc, const Texo<float2> texvert, const Texo<int4> textri, /**/ float *av) {
    dim3 avThreads(256, 1);
    dim3 avBlocks(1, nc);
    int nt, nv;
    nt = RBCnt;
    nv = RBCnv;
    Dzero(av, 2*nc);
    KL(dev::area_volume, (avBlocks, avThreads), (nt, nv, texvert, textri, av));
}

void area_volume_hst(int nc, const Texo<float2> texvert, const Texo<int4> textri, /**/ float *hst) {
    float *dev;
    Dalloc(&dev, 2*nc);
    area_volume(nc, texvert, textri, dev);
    cD2H(hst, dev, 2*nc);
    Dfree(dev);
}
