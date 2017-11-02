struct Cloud {
    const float *pp;
    const int *cc;   /* colors */
};

static void ini_cloud(Particle *pp, Cloud *c) {
    c->pp = (float*)pp;
};

static void ini_cloud_color(int *cc, Cloud *c) {
    c->cc = cc;
};
