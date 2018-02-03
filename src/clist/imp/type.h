enum {MAXA=2};                                /* maximum number of array */

// tag::map[]
struct ClistMap {
    int nA;              /* number of source arrays to build the cell lists, e.g remote+bulk -> 2 */
    uchar4 *ee[MAXA];    /* cell entries */
    uint *ii;            /* codes containing: indices of data to fetch and array id from which to fetch */
    Scan *scan;      /* scan workspace */
    long maxp;           /* maximum number of particles per input vector */
};
// end::map[]

typedef Sarray<uchar4*, MAXA> uchar4pA;       /* uchar4 pointers array               */
typedef Sarray<int, MAXA> intA;               /* int array                           */
typedef Sarray<Particle *, MAXA> ParticlepA;  /* particle pointers array             */
typedef Sarray<int *, MAXA> intpA;            /* particle pointers array             */
