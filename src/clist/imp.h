enum {MAXA=2};                                /* maximum number of array */

// tag::clist[]
struct Clist {
    int3 dims;
    int ncells;
    int *starts, *counts;
};
// end::clist[]

// tag::map[]
struct ClistMap {
    int nA;              /* number of source arrays to build the cell lists, e.g remote+bulk -> 2 */
    uchar4 *ee[MAXA];    /* cell entries */
    uint *ii;            /* codes containing: indices of data to fetch and array id from which to fetch */
    scan::Work scan;     /* scan workspace */
    long maxp;           /* maximum number of particles per input vector */
};
// end::map[]

void clist_ini(int LX, int LY, int LZ, /**/ Clist *c);
void clist_fin(/**/ Clist *c);

void clist_ini_map(int maxp, int nA, const Clist *c, /**/ ClistMap *m);
void clist_fin_map(ClistMap *m);

void clist_ini_counts(Clist *c);
void clist_subindex(bool project, int aid, int n, const PartList lp, /**/ Clist *c, ClistMap *m);
void clist_build_map(const int nn[], /**/ Clist *c, ClistMap *m);

/* special for fluid distribution */
void clist_subindex_local(int n, const PartList lp, /**/ Clist *c, ClistMap *m);
void clist_subindex_remote(int n, const PartList lp, /**/ Clist *c, ClistMap *m);

void clist_gather_pp(const Particle *pplo, const Particle *ppre, const ClistMap *m, long nout, /**/ Particle *ppout);
void clist_gather_ii(const int *iilo, const int *iire, const ClistMap *m, long nout, /**/ int *iiout);

/* quick cell build for single array */
void clist_build(int nlo, int nout, const Particle *pplo, /**/ Particle *ppout, Clist *c, ClistMap *m);
