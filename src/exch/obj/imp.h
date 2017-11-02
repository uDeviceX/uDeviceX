namespace exch {
namespace obj {

void ini(int nw, int maxd, int maxpsolid, Pack *p);
void ini(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ Comm *c);
void ini(int maxd, int maxpsolid, Unpack *u);
void ini(int maxd, int maxpsolid, PackF *p);
void ini(int maxd, int maxpsolid, UnpackF *u);

void fin(Pack *p);
void fin(Comm *c);
void fin(Unpack *u);
void fin(PackF *p);
void fin(UnpackF *u);

void build_map(int nw, const PaWrap *ww, /**/ Pack *p);
void pack(int nw, const PaWrap *ww, /**/ Pack *p);
void download(int nw, Pack *p);

void post_recv(Comm *c, Unpack *u);
void post_send(Pack *p, Comm *c);
void wait_recv(Comm *c, Unpack *u);
void wait_send(Comm *c);

int26 get_counts(Unpack *u);
Pap26 upload_shift(Unpack *u);
Fop26 reini_ff(const Unpack *u, PackF *pf);

void download_ff(PackF *p);

void post_recv_ff(Comm *c, UnpackF *u);
void post_send_ff(PackF *p, Comm *c);
void wait_recv_ff(Comm *c, UnpackF *u);
void wait_send_ff(Comm *c);

void unpack_ff(const UnpackF *u, const Pack *p, int nw, /**/ FoWrap *ww);

} // obj
} // exch
