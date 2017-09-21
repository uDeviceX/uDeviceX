namespace exch {
namespace obj {

void build_map(int nw, const ParticlesWrap *ww, /**/ Pack *p);
void pack(int nw, const ParticlesWrap *ww, Map map, /**/ Pack *p);
void download(Pack *p);

void post_recv(Comm *c, Unpack *u);
void post_send(Pack *p, Comm *c);
void wait_recv(Comm *c, Unpack *u);
void wait_send(Comm *c);

void upload(Unpack *u);
void download_ff();

void post_recv_ff(Comm *c, UnpackF *u);
void post_send_ff(PackF *p, Comm *c);
void wait_recv_ff(Comm *c, UnpackF *u);
void wait_send_ff(Comm *c);

void unpack_ff();

} // obj
} // exch
