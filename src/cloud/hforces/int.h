namespace hforces {
inline void ini_cloud(Particle *pp, Cloud *c) {
    c->pp = (float*)pp;
};
inline void ini_cloud(Particle *pp, int *cc, Cloud *c) {
    ini_cloud(pp, c);
    c->cc = cc;
};
} /* namespace */
