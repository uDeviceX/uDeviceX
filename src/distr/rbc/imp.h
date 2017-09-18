namespace distr {
namespace rbc {

void ini(int nv, Pack *p);
void ini(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ Comm *c);
void ini(int nv, Unpack *u);

void fin(Pack *p);
void fin(Comm *c);
void fin(Unpack *u);

void build_map(int nc, int nv, const Particle *pp, Pack *p);
void pack_pp(int nc, int nv, const Particle *pp, /**/ Pack *p);
void download(Pack *p);

void post_recv(Comm *c, Unpack *u);
void post_send(Pack *p, Comm *c);
void wait_recv(Comm *c, Unpack *u);
void wait_send(Comm *c);

using namespace ::rbc;
void unpack_bulk(const Pack *p, /**/ rbc::Quants *q);
void unpack_halo(const Unpack *u, /**/ rbc::Quants *q);

} // rbc
} // distr
