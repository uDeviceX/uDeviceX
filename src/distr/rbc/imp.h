namespace distr {
namespace rbc {

void build_map(int nc, int nv, const Particle *pp, Pack *p);
void pack_pp(int nc, int nv, const Particle *pp, /**/ Pack *p);
void download(Pack *p);

void post_recv(Comm *c, Unpack *u);
void post_send(Pack *p, Comm *c);
void wait_recv(Comm *c, Unpack *u);
void wait_send(Comm *c);

} // rbc
} // distr
