struct ParticlesWrap0 {
    const Particle *p;
    Force *f;
    int n;
    ParticlesWrap0() : p(NULL), f(NULL), n(0) {}
    ParticlesWrap0(const Particle *const p, const int n, Force *f)
        : p(p), n(n), f(f) {}
};
