namespace clist {

struct Clist {
    int3 dims;
    int ncells;
    int *starts, *counts;
};

struct Map {
    uchar4 *eelo, *eere; /* cell entries */
    uint *ii;
    scan::Work scan;
};

void ini(int LX, int LY, int LZ, /**/ Clist *c);
void fin(/**/ Clist *c);

void ini_ticket(const Clist *c, /**/ Map *t);
void fin_ticket(Map *t);


void ini_counts(Clist *c);
void subindex_local(int n, const Particle *pp, /**/ Clist *c, Map *t);
void subindex_remote(int n, const Particle *pp, /**/ Clist *c, Map *t);
void build_map(int nlo, int nre, /**/ Clist *c, Map *t);
void gather_pp(const Particle *pplo, const Particle *ppre, const Map *t, int nout, /**/ Particle *ppout);
void gather_ii(const int *iilo, const int *iire, const Map *t, int nout, /**/ int *iiout);

void build(int nlo, int nout, const Particle *pplo, /**/ Particle *ppout, Clist *c, Map *t);
void build(int nlo, int nre, int nout, const Particle *pplo, const Particle *ppre, /**/ Particle *ppout, Clist *c, Map *t);

} /* namespace */
