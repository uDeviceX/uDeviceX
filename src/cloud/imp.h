struct Cloud {
    const float *pp;
    const int *cc;   /* colors */
};

static void ini_cloud(const Particle *pp, Cloud *c) {
    c->pp = (const float*)pp;
};

static void ini_cloud_color(const int *cc, Cloud *c) {
    c->cc = cc;
};
