namespace comm {

typedef void data_t

enum {
    NFRAGS = 26, /* number of fragments for one halo */
    BULK   = 26, /* bulk id                          */
    NBAGS  = 27, /* fragments + bulk                 */
};

struct Bags {
    data_t *dev[NBAGS]; /* data on the device         */
    data_t *hst[NBAGS]; /* data on the host           */
    int  counts[NBAGS]; /* size of the data           */
    size_t bsize;       /* size of one datum in bytes */
};

struct Stamp {
    MPI_Request req[NBAGS]; /* requests */
    int bt;                 /* base tag */
    MPI_Comm cart;
    int ranks[NFRAGS], aranks[NFRAGS];
};

void post_recv(Bags *b, Stamp *s);
void post_send(Bags *b, Stamp *s);
void wait_all(Stamp *s);

} // comm
