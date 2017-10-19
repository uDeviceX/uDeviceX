namespace clist {

enum {MAXA=2};                                /* maximum number of array */
typedef Sarray<uchar4*, MAXA> uchar4pA;       /* uchar4 pointers array               */
typedef Sarray<int, MAXA> intA;               /* int array                           */
typedef Sarray<Particle *, MAXA> ParticlepA;  /* particle pointers array             */
typedef Sarray<int *, MAXA> intpA;            /* particle pointers array             */ 

struct Clist {
    int3 dims;
    int ncells;
    int *starts, *counts;
};

struct Map {
    uchar4 *eelo, *eere; /* cell entries */
    uint *ii;            /* codes containing: indices of data to fetch and array id from which to fetch */
    scan::Work scan;     /* scan workspace */
};

void ini(int LX, int LY, int LZ, /**/ Clist *c);
void fin(/**/ Clist *c);

void ini_map(const Clist *c, /**/ Map *m);
void fin_map(Map *m);


void ini_counts(Clist *c);
void subindex_local(int n, const Particle *pp, /**/ Clist *c, Map *m);
void subindex_remote(int n, const Particle *pp, /**/ Clist *c, Map *m);
void build_map(int nlo, int nre, /**/ Clist *c, Map *m);
void gather_pp(const Particle *pplo, const Particle *ppre, const Map *m, int nout, /**/ Particle *ppout);
void gather_ii(const int *iilo, const int *iire, const Map *m, int nout, /**/ int *iiout);

void build(int nlo, int nout, const Particle *pplo, /**/ Particle *ppout, Clist *c, Map *m);

} /* namespace */
