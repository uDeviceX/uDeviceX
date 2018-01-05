namespace distr {
namespace rig {

// tag::interface[]
void ini(int maxns, int nv, Pack *p);
void ini(MPI_Comm comm, /**/ Comm *c);
void ini(int maxns, int nv, Unpack *u);

void fin(Pack *p);
void fin(Comm *c);
void fin(Unpack *u);

void build_map(int ns, const Solid *ss, /**/ Pack *p);
void pack(int ns, int nv, const Solid *ss, const Particle *ipp, /**/ Pack *p);
void download(Pack *p);

void post_recv(Comm *c, Unpack *u);
void post_send(Pack *p, Comm *c);
void wait_recv(Comm *c, Unpack *u);
void wait_send(Comm *c);

using namespace ::rig;
void unpack_bulk(const Pack *p, /**/ rig::Quants *q);
void unpack_halo(const Unpack *u, /**/ rig::Quants *q);
// end::interface[]

} // rig
} // distr
