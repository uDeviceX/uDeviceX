namespace odstr {
namespace sub {

void waitall_s(MPI_Request *reqs);
void waitall_r(MPI_Request *reqs);

void post_recv(const int rank[], const int btc, const int btp,
               MPI_Request *size_req, MPI_Request *mesg_req, Recv *r);

void post_recv_ii(const int rank[], const int tags[], const int bt, /**/ MPI_Request *ii_req, Pbufs<int> *rii);

void halo(const Particle *pp, int n, Send *s);
void scan(int n, Send *s);

void pack_pp(const Particle *pp, int n, Send *s);
void pack_ii(const int *ii, int n, const Send *s, Pbufs<int>* sii);

void send_sz(const int rank[], const int btc, /**/ Send *s, MPI_Request *req);
void send_pp(const int rank[], const int btp, /**/ Send *s, MPI_Request *req);
void send_ii(const int rank[], const int size[], const int bt, /**/ Pbufs<int> *sii, MPI_Request *req);

void count(/**/ Recv *r, int *nhalo);
int count_sz(Send *s);

void unpack_pp(const int n, const Recv *r, /**/ Particle *pp_re);
void unpack_ii(const int n, const Recv *r, const Pbufs<int> *rii, /**/ int *ii_re);

void subindex(const int n, const Recv *r, /*io*/ Particle *pp_re, int *counts, /**/ uchar4 *subi);

/* TODO: this is not used, why? */
void cancel_recv(/**/ MPI_Request *size_req, MPI_Request *mesg_req);

void scatter(bool remote, const uchar4 *subi, const int n, const int *start, /**/ uint *iidx);
void gather_id(const int *ii_lo, const int *ii_re, int n, const uint *iidx, /**/ int *ii);
void gather_pp(const float2  *pp_lo, const float2 *pp_re, int n, const uint *iidx,
               /**/ float2  *pp, float4  *zip0, ushort4 *zip1);

// ini.h
void ini_comm(/**/ int rank[], int tags[]);
void ini_S(/**/ Send *s);
void ini_R(const Send *s, /**/ Recv *r);
void ini_SRI(Pbufs<int> *sii, Pbufs<int> *rii);

// fin.h
void fin_S(Send *s);
void fin_R(Recv *r);
void fin_SRI(Pbufs<int> *sii, Pbufs<int> *rii);
}
}
