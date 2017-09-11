struct ParticlesWrap {
    const Particle *p;
    int n;
    Force *f;
    ParticlesWrap() : p(NULL), n(0), f(NULL)  {}
    ParticlesWrap(const Particle *const p, const int n, Force *f)
        : p(p), n(n), f(f) {}
};

 /* [pa]ritcle and [fo]rce [p]ointers ;  see also dpdr/ */
typedef Sarray<Particle*, 26> Pap26;
typedef Sarray<Force*,    26> Fop26;
