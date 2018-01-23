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
    data_t *data[NBAGS]; /* data on the host                    */
    int         *counts; /* size of the data                    */
    int capacity[NBAGS]; /* capacity of each frag (elem number) */
    size_t bsize;        /* size of one datum in bytes          */
};
// end::hBags[]

struct Comm;

// tag::alloc[]
int comm_bags_ini(AllocMod fmod, AllocMod bmod, size_t bsize, const int capacity[NBAGS], /**/ hBags *hb, dBags *db);
int comm_bags_fin(AllocMod fmod, AllocMod bmod, /**/ hBags *hb, dBags *db);
int comm_ini(MPI_Comm cart, /**/ Comm **c);
int comm_fin(/**/ Comm *c);
// end::alloc[]

// tag::communication[]
int post_recv(hBags *b, Comm *c);           // <1>
int post_send(const hBags *b, Comm *c);     // <2>

int wait_recv(Comm *c, /**/ hBags *b);      // <3>
int wait_send(Comm *c);                     // <4>
// end::communication[]

int    comm_get_number_capacity(int i, const hBags *b);
size_t comm_get_byte_capacity(int i, const hBags *b);
