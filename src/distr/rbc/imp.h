namespace distr {
namespace rbc {

// tag::interface[]
void drbc_pack_ini(int maxc, int nv, Pack *p);
void drbc_comm_ini(MPI_Comm comm, /**/ Comm *c);
void drbc_unpack_ini(int maxc, int nv, Unpack *u);

void drbc_pack_fin(Pack *p);
void drbc_comm_fin(Comm *c);
void drbc_unpack_fin(Unpack *u);

using namespace ::rbc;

void drbc_build_map(int nc, int nv, const Particle *pp, Pack *p);
void drbc_pack(const rbc::Quants *q, /**/ Pack *p);
void drbc_download(Pack *p);

void drbc_post_recv(Comm *c, Unpack *u);
void drbc_post_send(Pack *p, Comm *c);
void drbc_wait_recv(Comm *c, Unpack *u);
void drbc_wait_send(Comm *c);

void drbc_unpack_bulk(const Pack *p, /**/ rbc::Quants *q);
void drbc_unpack_halo(const Unpack *u, /**/ rbc::Quants *q);
// end::interface[]

} // rbc
} // distr
