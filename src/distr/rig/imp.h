namespace distr {
namespace rig {

// tag::interface[]
void drig_pack_ini(int maxns, int nv, Pack *p);
void drig_comm_ini(MPI_Comm comm, /**/ Comm *c);
void drig_unpack_ini(int maxns, int nv, Unpack *u);

void drig_pack_fin(Pack *p);
void drig_comm_fin(Comm *c);
void drig_unpack_fin(Unpack *u);

void drig_build_map(int ns, const Solid *ss, /**/ Pack *p);
void drig_pack(int ns, int nv, const Solid *ss, const Particle *ipp, /**/ Pack *p);
void drig_download(Pack *p);

void drig_post_recv(Comm *c, Unpack *u);
void drig_post_send(Pack *p, Comm *c);
void drig_wait_recv(Comm *c, Unpack *u);
void drig_wait_send(Comm *c);

using namespace ::rig;
void drig_unpack_bulk(const Pack *p, /**/ rig::Quants *q);
void drig_unpack_halo(const Unpack *u, /**/ rig::Quants *q);
// end::interface[]

} // rig
} // distr
