namespace exch {
namespace obj {

void ini(int nw, int maxd, Pack *p);
void ini(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ Comm *c);
void ini(int maxd, Unpack *u);
void ini(int maxd, PackF *p);
void ini(int maxd, UnpackF *u);

void fin(Pack *p);
void fin(Comm *c);
void fin(Unpack *u);
void fin(PackF *p);
void fin(UnpackF *u);

void build_map(int nw, const ParticlesWrap *ww, /**/ Pack *p);
void pack(int nw, const ParticlesWrap *ww, Map map, /**/ Pack *p);
Pap26 download(Pack *p);

void post_recv(Comm *c, Unpack *u);
void post_send(Pack *p, Comm *c);
void wait_recv(Comm *c, Unpack *u);
void wait_send(Comm *c);

void upload(Unpack *u);
Fop26 reini_ff(const Pack *p, PackF *pf);

void download_ff(PackF *p);

void post_recv_ff(Comm *c, UnpackF *u);
void post_send_ff(PackF *p, Comm *c);
void wait_recv_ff(Comm *c, UnpackF *u);
void wait_send_ff(Comm *c);

void unpack_ff();

} // obj
} // exch
