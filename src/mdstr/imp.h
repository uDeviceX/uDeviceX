namespace mdstr {
namespace sub {

// ini
void gen_ne(MPI_Comm cart, /**/ int* rnk_ne, int* ank_ne);

// imp
void get_dests(const float *rr, int nm, /**/ int *dests[27], int counts[27]);

void pack(int *dests[27], const int counts[27], const Particle *pp, int nv, /**/ Particle *pps[27]);

void post_send(int nv, const int counts[27], Particle *const pp[27], MPI_Comm cart, int btc, int btp, int rnk_ne[27],
               /**/ MPI_Request sreqc[26], MPI_Request sreqp[26]);

void post_recv(MPI_Comm cart, int btc, int btp, int ank_ne[27],
               /**/ int counts[27], Particle *pp[27], MPI_Request rreqc[26], MPI_Request rreqp[26]);

int unpack(int nv, Particle *const ppr[27], const int counts[27], /**/ Particle *pp);

} // sub
} // mdstr
