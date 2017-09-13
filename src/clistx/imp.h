namespace clist {

struct Clist {
    int3 dims;
    int ncells;
    int *starts, *counts;
};

struct Work {
    uchar4 *eelo, *eere; /* cell entries */
    uint *ii;
    scan::Work scan;
};

void ini(int LX, int LY, int LZ, /**/ Clist *c);
void fin(/**/ Clist *c);

void ini_work(const Clist *c, /**/ Work *w);
void fin_work(Work *w);

void build(int nlo, int nout, const Particle *pplo, /**/ Particle *ppout, Clist *c, /*w*/ Work *w);
void build(int nlo, int nre, int nout, const Particle *pplo, const Particle *ppre, /**/ Particle *ppout, Clist *c, /*w*/ Work *w);

} /* namespace */
