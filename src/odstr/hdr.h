class Distr {
public:
    void cancel_recv(MPI_Request *size_req, MPI_Request *mesg_req);

    /* decl */
    Send s;
    Recv r;
};
