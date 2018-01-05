namespace distr {
namespace rbc {

// tag::interface[]
void ini(int maxc, int nv, Pack *p);
void ini(MPI_Comm comm, /**/ Comm *c);
void ini(int maxc, int nv, Unpack *u);

void fin(Pack *p);
void fin(Comm *c);
void fin(Unpack *u);

using namespace ::rbc;

void build_map(int nc, int nv, const Particle *pp, Pack *p);
void pack(const rbc::Quants *q, /**/ Pack *p);
void download(Pack *p);

void post_recv(Comm *c, Unpack *u);
void post_send(Pack *p, Comm *c);
void wait_recv(Comm *c, Unpack *u);
void wait_send(Comm *c);

void unpack_bulk(const Pack *p, /**/ rbc::Quants *q);
void unpack_halo(const Unpack *u, /**/ rbc::Quants *q);
// end::interface[]

} // rbc
} // distr
