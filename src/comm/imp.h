namespace comm {

typedef void data_t;

enum {
    NFRAGS = 26, /* number of fragments for one halo */
    BULK   = 26, /* bulk id                          */
    NBAGS  = 27, /* fragments + bulk                 */
};

/* Allocation mod for bags */
enum AllocMod {
    HST_ONLY,  /* only host bags allocated    */
    DEV_ONLY,  /* only device bags allocated  */
    PINNED,    /* both host and device pinned */
    NONE       /* no allocation               */
};

struct dBags {
    data_t *data[NBAGS]; /* data on the device         */
};

struct hBags {
    data_t *data[NBAGS]; /* data on the host           */
    int         *counts; /* size of the data           */
    int capacity[NBAGS]; /* capacity of each frag      */
    size_t bsize;        /* size of one datum in bytes */
};

struct Stamp {
    MPI_Request sreq[NBAGS]; /* send requests */
    MPI_Request rreq[NBAGS]; /* recv requests */
    int bt;                  /* base tag */
    MPI_Comm cart;
    int ranks[NFRAGS];       /* ranks of neighbors     */
    int  tags[NFRAGS];       /* tags in bt coordinates */
};

void ini(AllocMod fmod, AllocMod bmod, size_t bsize, const int capacity[NBAGS], /**/ hBags *hb, dBags *db);
void fin(AllocMod fmod, AllocMod bmod, /**/ hBags *hb, dBags *db);

/* stamp alloc */
void ini(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ Stamp *s);
void fin(/**/ Stamp *s);

/* communication */
void post_recv(hBags *b, Stamp *s);
void post_send(hBags *b, Stamp *s);

void wait_recv(Stamp *s, /**/ hBags *b);
void wait_send(Stamp *s);

} // comm
