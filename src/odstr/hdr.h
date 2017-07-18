class Distr {
public:
    void recv_count(int *nhalo);
    void unpack_pp(int n, /*o*/ Particle *pp_re);
    void unpack_ii(int n, /*o*/ int *ii_re);
    void subindex_remote(int n, /*io*/ Particle *pp_re, int *counts, /**/ uchar4 *subi);
    void cancel_recv(MPI_Request *size_req, MPI_Request *mesg_req);

    /* decl */
    Send s;
    Recv r;
};
