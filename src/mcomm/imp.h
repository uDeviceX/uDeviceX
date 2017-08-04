namespace mcomm {
namespace sub {

// ini

void ini_tcom(MPI_Comm cart, /**/ MPI_Comm *newcart, int rnk_ne[27], int ank_ne[27]);

// fin

void fin_tcom(const bool first, /**/ MPI_Comm *cart, Reqs *sreq, Reqs *rreq);

// imp

void cancel_req(Reqs *r);
void wait_req(Reqs *r);
int map(const float3* minext_hst, const float3 *maxext_hst, const int nm, /**/ std::vector<int> travellers[27], int counts[27]);
void pack(const Particle *pp, const int nv, const std::vector<int> travellers[27], /**/ Particle *spp[27]);
void post_recv(MPI_Comm cart, const int ank_ne[26], int btc, int btp, /**/ int counts[27], Particle *pp[27], Reqs *rreqs);
void post_send(MPI_Comm cart, const int rnk_ne[26], int btc, int btp, int nv, const int counts[27], const Particle *const pp[27], /**/ Reqs *sreqs);
int unpack(const int counts[27], const Particle *const rpp[27], const int nv, /**/ Particle *pp);

} // sub
} // mcomm
