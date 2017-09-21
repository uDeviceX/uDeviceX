struct ParticlesWrap {
    const Particle *p;
    int n;
    Force *f;
    ParticlesWrap() : p(NULL), n(0), f(NULL)  {}
    ParticlesWrap(const Particle *const p, const int n, Force *f)
        : p(p), n(n), f(f) {}
};
