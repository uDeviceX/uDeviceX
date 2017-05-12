namespace mbounce // [M]esh bounce
{
    void bounce_tcells_hst(const Force *ff, const Mesh m, const Particle *i_pp, const int *tcellstarts, const int *tcellcounts, const int *tids,
                           const int n, /**/ Particle *pp, Solid *ss);
}
