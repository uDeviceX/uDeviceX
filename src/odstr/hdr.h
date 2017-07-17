class Distr {
public:
    void ini(MPI_Comm cart, int rank[]);
    void fin();

    void waitall(MPI_Request *reqs);
    void post_recv(MPI_Comm cart, int rank[], MPI_Request *size_req, MPI_Request *mesg_req);
    void post_recv_ii(MPI_Comm cart, int rank[], MPI_Request *ii_req);
    void halo(Particle *pp, int n);
    void scan(int n);
    void pack_pp(const Particle *pp, int n);
    void pack_ii(const int *ii, int n);
    int  send_sz(MPI_Comm cart, int rank[], MPI_Request *req);
    void send_pp(MPI_Comm cart, int rank[], MPI_Request *req);
    void send_ii(MPI_Comm cart, int rank[], MPI_Request *req);
    void recv_count(int *nhalo);
    void unpack_pp(int n, /*o*/ Particle *pp_re);
    void unpack_ii(int n, /*o*/ int *ii_re);
    void subindex_remote(int n, /*io*/ Particle *pp_re, int *counts, /**/ uchar4 *subi);
    void cancel_recv(MPI_Request *size_req, MPI_Request *mesg_req);

    /* decl */
    Send s;
    Recv r;
};
