namespace mbounce {

struct Momentum {
    float P[3], L[3]; /* linear and angular momentum */
};

struct Work {
    Momentum *mm_dev, *mm_hst;
};

void alloc_work(Work *w);
void free_work(Work *w);

void bounce_tcells_hst(const Force *ff, const Mesh m, const Particle *i_pp, const int *tcellstarts, const int *tcellcounts, const int *tids,
                       const int n, /**/ Particle *pp, Solid *ss);
    
void bounce_tcells_dev(const Force *ff, const Mesh m, const Particle *i_pp, const int *tcellstarts, const int *tcellcounts, const int *tids,
                       const int n, /**/ Particle *pp, Solid *ss);

} // mbounce
