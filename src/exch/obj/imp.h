namespace exch {
namespace obj {

void eobj_pack_ini(int nw, int maxd, int maxpsolid, Pack *p);
void eobj_comm_ini(MPI_Comm comm, /**/ Comm *c);
void eobj_unpack_ini(int maxd, int maxpsolid, Unpack *u);
void eobj_packf_ini(int maxd, int maxpsolid, PackF *p);
void eobj_unpackf_ini(int maxd, int maxpsolid, UnpackF *u);

void eobj_pack_fin(Pack *p);
void eobj_comm_fin(Comm *c);
void eobj_unpack_fin(Unpack *u);
void eobj_packf_fin(PackF *p);
void eobj_unpackf_fin(UnpackF *u);

void eobj_build_map(int nw, const PaWrap *ww, /**/ Pack *p);
void eobj_pack(int nw, const PaWrap *ww, /**/ Pack *p);
void eobj_download(int nw, Pack *p);

void eobj_post_recv(Comm *c, Unpack *u);
void eobj_post_send(Pack *p, Comm *c);
void eobj_wait_recv(Comm *c, Unpack *u);
void eobj_wait_send(Comm *c);

int26 eobj_get_counts(Unpack *u);
Pap26 eobj_upload_shift(Unpack *u);
Fop26 eobj_reini_ff(const Unpack *u, PackF *pf);

void eobj_download_ff(PackF *p);

void eobj_post_recv_ff(Comm *c, UnpackF *u);
void eobj_post_send_ff(PackF *p, Comm *c);
void eobj_wait_recv_ff(Comm *c, UnpackF *u);
void eobj_wait_send_ff(Comm *c);

void eobj_unpack_ff(const UnpackF *u, const Pack *p, int nw, /**/ FoWrap *ww);

} // obj
} // exch
