namespace comm {

typedef void data_t;

enum {
    NFRAGS = 26, /* number of fragments for one halo */
    BULK   = 26, /* bulk id                          */
    NBAGS  = 27, /* fragments + bulk                 */
};

struct dBags {
    data_t *data[NBAGS]; /* data on the device         */
    int         *counts; /* size of the data           */
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

/* pinned allocation */
void ini_pinned_no_bulk(size_t bsize, float maxdensity, /**/ hBags *hb, dBags *db);
void ini_pinned_full(size_t bsize, float maxdensity, /**/ hBags *hb, dBags *db);
void fin_pinned(/**/ hBags *hb, dBags *db);

/* normal allocation */
void ini_no_bulk(size_t bsize, float maxdensity, /**/ hBags *hb, dBags *db);
void ini_full(size_t bsize, float maxdensity, /**/ hBags *hb, dBags *db);
void fin(/**/ hBags *hb, dBags *db);

/* normal allocation host only */
void ini_no_bulk(size_t bsize, float maxdensity, /**/ hBags *hb);
void ini_full(size_t bsize, float maxdensity, /**/ hBags *hb);
void fin(/**/ hBags *hb);

/* stamp alloc */
void ini(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ Stamp *s);
void fin(/**/ Stamp *s);

/* communication */
void post_recv(hBags *b, Stamp *s);
void post_send(hBags *b, Stamp *s);

void wait_recv(Stamp *s, /**/ hBags *b);
void wait_send(Stamp *s);

} // comm
