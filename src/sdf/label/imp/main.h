void dev(Sdf *sdf, int n, const Particle *pp, /**/ int *labels) {
    Sdf_v sdf_v;
    sdf_to_view(sdf, &sdf_v);
    KL(dev0::main,(k_cnf(n)), (sdf_v, n, pp, labels));
}

void hst(Sdf *sdf, int n, const Particle *pp, /**/ int *hst) {
    int *dev;
    Sdf_v sdf_v;
    Dalloc(&dev, n);
    sdf_to_view(sdf, &sdf_v);
    KL(dev0::main,(k_cnf(n)), (sdf_v, n, pp, dev));
    cD2H(hst, dev, n);
    Dfree(dev);
}
