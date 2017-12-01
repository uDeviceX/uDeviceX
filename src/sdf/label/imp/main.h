void dev(const sdf::tex3Dca texsdf, int n, const Particle *pp, /**/ int *labels) {
    KL(dev0::main,(k_cnf(n)), (texsdf, n, pp, labels));
}

void hst(const sdf::tex3Dca texsdf, int n, const Particle *pp, /**/ int *hst) {
    int *dev;
    Dalloc(&dev, n);
    KL(dev0::main,(k_cnf(n)), (texsdf, n, pp, dev));
    cD2H(hst, dev, n);
    Dfree(dev);
}
