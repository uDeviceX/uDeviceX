void area_volume(int nc, const Texo<float2> texvert, const Texo<int4> textri, /**/ float *av) {
    dim3 avThreads(256, 1);
    dim3 avBlocks(1, nc);
    int nt, nv;
    nt = RBCnt;
    nv = RBCnv;
    Dzero(av, 2*nc);
    KL(dev::area_volume, (avBlocks, avThreads), (nt, nv, texvert, textri, av));
}
