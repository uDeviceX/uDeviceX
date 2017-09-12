namespace comm {

typedef void data_t;

enum {
    NFRAGS = 26, /* number of fragments for one halo */
    BULK   = 26, /* bulk id                          */
    NBAGS  = 27, /* fragments + bulk                 */
};

struct Bags {
    data_t  *dev[NBAGS]; /* data on the device         */
    data_t  *hst[NBAGS]; /* data on the host           */
    int   counts[NBAGS]; /* size of the data           */
    int capacity[NBAGS]; /* capacity of each frag      */
    size_t bsize;        /* size of one datum in bytes */
};

struct Stamp {
    MPI_Request sreq[NBAGS]; /* send requests */
    MPI_Request rreq[NBAGS]; /* recv requests */
    int bt;                  /* base tag */
    MPI_Comm cart;
    int rnks[NFRAGS];        /* ranks of neighbors      */
    int anks[NFRAGS];        /* anti ranks of neighbors */
};

void ini_no_bulk(size_t bsize, float maxdensity, /**/ Bags *b);
void ini_full   (size_t bsize, float maxdensity, /**/ Bags *b);

void fin(/**/ Bags *b);

void ini(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ Stamp *s);
void fin(/**/ Stamp *s);

void post_recv(Bags *b, Stamp *s);
void post_send(Bags *b, Stamp *s);
void recv_counts(const Stamp *s, /**/ Bags *b);
void wait_recv(Stamp *s);
void wait_send(Stamp *s);

} // comm
