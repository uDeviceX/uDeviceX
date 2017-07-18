class Distr {
public:
    void subindex_remote(int n, /*io*/ Particle *pp_re, int *counts, /**/ uchar4 *subi);
    void cancel_recv(MPI_Request *size_req, MPI_Request *mesg_req);

    /* decl */
    Send s;
    Recv r;
};
