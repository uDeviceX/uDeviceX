void bounce_back(Wvel_d wv, Coords c, const tex3Dca texsdf, int n, /**/ Particle *pp) {
    KL(dev::bounce_back, (k_cnf(n)), (wv, c, texsdf, n, /**/ pp));
}
