namespace comm {

typedef void data_t;

enum {
    NFRAGS = 26, /* number of fragments for one halo */
    BULK   = 26, /* bulk id                          */
    NBAGS  = 27, /* fragments + bulk                 */
};

/* Allocation mod for bags */
enum AllocMod {
    // tag::AllocMod[]
    HST_ONLY,   /* only host bags allocated                 */
    DEV_ONLY,   /* only device bags allocated               */
    PINNED,     /* both host and device pinned              */
    PINNED_HST, /* host pinned; no device memory            */
    PINNED_DEV, /* host pinned; device global memory on gpu */
    NONE        /* no allocation                            */
    // end::AllocMod[]
};

// tag::dBags[]
struct dBags {
    data_t *data[NBAGS]; /* data on the device         */
};
// end::dBags[]

// tag::hBags[]
struct hBags {
    data_t *data[NBAGS]; /* data on the host           */
    int         *counts; /* size of the data           */
    int capacity[NBAGS]; /* capacity of each frag      */
    size_t bsize;        /* size of one datum in bytes */
};
// end::hBags[]

// tag::stamp[]
struct Stamp {
    MPI_Request sreq[NBAGS]; /* send requests */
    MPI_Request rreq[NBAGS]; /* recv requests */
    MPI_Comm cart;           /* cartesian communicator */
    int ranks[NFRAGS];       /* ranks of neighbors     */
    int  tags[NFRAGS];       /* tags in bt coordinates */
};
// end::stamp[]

// tag::alloc[]
/* bags alloc */
int bags_ini(AllocMod fmod, AllocMod bmod, size_t bsize, const int capacity[NBAGS], /**/ hBags *hb, dBags *db);
int bags_fin(AllocMod fmod, AllocMod bmod, /**/ hBags *hb, dBags *db);

/* stamp alloc */
int ini(MPI_Comm comm, /**/ Stamp *s);
int fin(/**/ Stamp *s);
// end::alloc[]

// tag::communication[]
int post_recv(hBags *b, Stamp *s);           // <1>
int post_send(const hBags *b, Stamp *s);     // <2>

int wait_recv(Stamp *s, /**/ hBags *b);      // <3>
int wait_send(Stamp *s);                     // <4>
// end::communication[]

} // comm
