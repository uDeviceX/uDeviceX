namespace hforces {
inline void ini_cloud(Particle *pp, Cloud *c) {
    c->pp = (float*)pp;
};
inline void ini_cloud_color(int *cc, Cloud *c) {
    c->cc = cc;
};
} /* namespace */
