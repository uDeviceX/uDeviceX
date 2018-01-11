namespace exch {
namespace obj {

void eobj_pack_ini(int nw, int maxd, int maxpsolid, EObjPack *p);
void eobj_comm_ini(MPI_Comm comm, /**/ EObjComm *c);
void eobj_unpack_ini(int maxd, int maxpsolid, EObjUnpack *u);
void eobj_packf_ini(int maxd, int maxpsolid, EObjPackF *p);
void eobj_unpackf_ini(int maxd, int maxpsolid, EObjUnpackF *u);

void eobj_pack_fin(EObjPack *p);
void eobj_comm_fin(EObjComm *c);
void eobj_unpack_fin(EObjUnpack *u);
void eobj_packf_fin(EObjPackF *p);
void eobj_unpackf_fin(EObjUnpackF *u);

void eobj_build_map(int nw, const PaWrap *ww, /**/ EObjPack *p);
void eobj_pack(int nw, const PaWrap *ww, /**/ EObjPack *p);
void eobj_download(int nw, EObjPack *p);

void eobj_post_recv(EObjComm *c, EObjUnpack *u);
void eobj_post_send(EObjPack *p, EObjComm *c);
void eobj_wait_recv(EObjComm *c, EObjUnpack *u);
void eobj_wait_send(EObjComm *c);

int26 eobj_get_counts(EObjUnpack *u);
Pap26 eobj_upload_shift(EObjUnpack *u);
Fop26 eobj_reini_ff(const EObjUnpack *u, EObjPackF *pf);

void eobj_download_ff(EObjPackF *p);

void eobj_post_recv_ff(EObjComm *c, EObjUnpackF *u);
void eobj_post_send_ff(EObjPackF *p, EObjComm *c);
void eobj_wait_recv_ff(EObjComm *c, EObjUnpackF *u);
void eobj_wait_send_ff(EObjComm *c);

void eobj_unpack_ff(const EObjUnpackF *u, const EObjPack *p, int nw, /**/ FoWrap *ww);

} // obj
} // exch
