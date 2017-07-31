namespace rdstr {
namespace sub {

void waitall(MPI_Request rr[26]);
void cancelall(MPI_Request rr[26]);

void extents(const Particle *pp, int nc, int nv, /**/ float3 *ll, float3 *hh);
void get_pos(int nc, const float3 *ll, const float3 *hh, /**/ float *rr);

namespace gen = mdstr::gen;
typedef gen::pbuf<Particle> Partbuf;


void pack(int *reord[27], const int counts[27], const Particle *pp, int nv, /**/ Partbuf *bpp);

void post_send(int nv, const int counts[27], const Partbuf *bpp, MPI_Comm cart, int bt, int rnk_ne[27],
               /**/ MPI_Request sreq[26]);

void post_recv(MPI_Comm cart, int nmax, int bt, int ank_ne[27], /**/ Partbuf *bpp, MPI_Request rreq[26]);

int unpack(int npd, const Partbuf *bpp, const int counts[27], /**/ Particle *pp);

void shift(int npd, const int counts[27], /**/ Particle *pp);

} // sub
} // rdstr
