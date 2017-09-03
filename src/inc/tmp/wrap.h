struct ParticlesWrap {
    const Particle *p;
    Force *f;
    int n;
    ParticlesWrap() : p(NULL), f(NULL), n(0) {}
    ParticlesWrap(const Particle *const p, const int n, Force *f)
        : p(p), n(n), f(f) {}
};

 /* [pa]ritcle and [fo]rce [p]ointers ;  see also dpdr/ */
typdef Sarray<Particle*, 26> Pap26;
typdef Sarray<Force*,    26> Fop26;
