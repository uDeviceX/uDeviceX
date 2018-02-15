void wall_label_dev(const Sdf *sdf, int n, const Particle *pp, /**/ int *labels) {
    Sdf_v sdf_v;
    sdf_to_view(sdf, &sdf_v);
    KL(sdf_label_dev::main,(k_cnf(n)), (sdf_v, n, pp, labels));
}

void wall_label_hst(const Sdf *sdf, int n, const Particle *pp, /**/ int *hst) {
    int *dev;
    Sdf_v sdf_v;
    Dalloc(&dev, n);
    sdf_to_view(sdf, &sdf_v);
    KL(sdf_label_dev::main,(k_cnf(n)), (sdf_v, n, pp, dev));
    cD2H(hst, dev, n);
    Dfree(dev);
}
