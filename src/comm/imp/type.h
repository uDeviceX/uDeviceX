// tag::comm[]
struct Comm {
    MPI_Request sreq[NBAGS]; /* send requests */
    MPI_Request rreq[NBAGS]; /* recv requests */
    MPI_Comm cart;           /* cartesian communicator */
    int ranks[NFRAGS];       /* ranks of neighbors     */
    int  tags[NFRAGS];       /* tags in bt coordinates */
};
// end::comm[]

struct CommBuffer {
    data_t *buf;
    size_t sz;
};
