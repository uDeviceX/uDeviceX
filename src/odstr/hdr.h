struct Send {
    int **iidx; /* indices */
    float2 **dev;   /* buffers */
    float *hst[27];

    int *size_dev, *strt;
    int size[27];
    PinnedHostBuffer4<int>* size_pin;

    float2 *hst_[27];
    int    *iidx_[27];
};

struct Recv {
    float2 **dev;
    int *strt;
    int tags[27];
    int    size[27];
    float *hst[27];
    float2 *hst_[27];
};

class Distr {
public:
    void ini(MPI_Comm cart, int rank[]);
    void fin();

    void waitall(MPI_Request *reqs);
    void post_recv(MPI_Comm cart, int rank[], MPI_Request *size_req, MPI_Request *mesg_req);
    void halo(Particle *pp, int n);
    void scan(int n);
    void pack(Particle *pp, int n);
    int send_sz(MPI_Comm cart, int rank[], MPI_Request *req);
    void send_msg(MPI_Comm cart, int rank[], MPI_Request *req);
    void recv_count(int *nhalo);
    void unpack(int n_pa, /*io*/ int *count, /*o*/ uchar4 *subi, Particle *pp_re);
    void cancel_recv(MPI_Request *size_req, MPI_Request *mesg_req);
    /* decl */
    Send s;
    Recv r;
};
