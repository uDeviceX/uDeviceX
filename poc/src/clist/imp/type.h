// tag::clist[]
struct Clist {
    int3 dims;
    int ncells;
    int *starts, *counts;
};
// end::clist[]

// tag::map[]
struct ClistMap {
    int nA;       /* number of source arrays to build the cell lists, e.g remote+bulk -> 2 */
    uchar4 **ee;  /* cell entries */
    uint *ii;     /* codes containing: indices of data to fetch and array id from which to fetch */
    Scan *scan;   /* scan workspace */
    long maxp;    /* maximum number of particles per input vector */
};
// end::map[]
