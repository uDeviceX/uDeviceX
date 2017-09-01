namespace hforces {
inline void ini_cloudA(Particle *pp, CloudA *c) {
    c->pp = (float*)pp;
};
inline void ini_cloudB(Particle *pp, CloudB *c) {
    c->pp = (float2*)pp;
};
} /* namespace */
