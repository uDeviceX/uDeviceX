namespace hideme {
void waitall(MPI_Request *reqs);
void post_recv(MPI_Comm cart, int rank[], MPI_Request *size_req, MPI_Request *mesg_req);
void halo(Particle *pp, int n);
void scan(int n);
void pack(Particle *pp, int n);
int send_sz(MPI_Comm cart, int rank[], MPI_Request *req);
void send_msg(MPI_Comm cart, int rank[], MPI_Request *req);
void recv_count(int *nhalo_padded, int *nhalo);
void unpack(int n_pa, /*io*/ int *count, /*o*/ uchar4 *subi, Particle *pp_re);
void cancel_recv(MPI_Request *size_req, MPI_Request *mesg_req);
}
