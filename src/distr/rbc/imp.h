namespace distr {
namespace rbc {

void ini(float maxdensity, Pack *p);
void ini(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ Comm *c);
void ini(float maxdensity, Unpack *u);

void fin(Pack *p);
void fin(Comm *c);
void fin(Unpack *u);

void link_bulk(Pack *p, Unpack *U);

void build_map(int nc, int nv, const Particle *pp, Pack *p);
void pack_pp(int nc, int nv, const Particle *pp, /**/ Pack *p);
void download(Pack *p);

void post_recv(Comm *c, Unpack *u);
void post_send(Pack *p, Comm *c);
void wait_recv(Comm *c, Unpack *u);
void wait_send(Comm *c);

} // rbc
} // distr
