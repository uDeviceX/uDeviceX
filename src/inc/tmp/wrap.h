struct ParticlesWrap000 {
    const Particle *p;
    Force *f;
    int n;
    ParticlesWrap000() : p(NULL), f(NULL), n(0) {}
    ParticlesWrap000(const Particle *const p, const int n, Force *f)
        : p(p), n(n), f(f) {}
};
