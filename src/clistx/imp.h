namespace clist {

struct Clist {
    int3 dims;
    int ncells;
    int *start, *count;
};

struct Work {
    uchar4 *eelo, *eere; /* cell entries */
    int *ii;
    scan::Work ws;
};

void ini(int LX, int LY, int LZ, /**/ Clist *c);
void fin(/**/ Clist *c);

void ini_work(Work *w);
void fin_work(Work *w);

void build(const Particle *pplo, /**/ Particle *pp, Clist *c, /*w*/ Work *w);
void build(const Particle *pplo, const Particle *ppre, /**/ Particle *pp, Clist *c, /*w*/ Work *w);

} /* namespace */
