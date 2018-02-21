struct EObjPack;
struct EObjComm;
struct EObjUnpack;
struct EObjPackF;
struct EObjUnpackF;

// tag::mem[]
void eobj_pack_ini(int3 L, int nw, int maxd, int maxpsolid, EObjPack **p);
void eobj_comm_ini(MPI_Comm cart, /**/ EObjComm **c);
void eobj_unpack_ini(int3 L, int maxd, int maxpsolid, EObjUnpack **u);
void eobj_packf_ini(int3 L, int maxd, int maxpsolid, EObjPackF **p);
void eobj_unpackf_ini(int3 L, int maxd, int maxpsolid, EObjUnpackF **u);
// end::mem[]

// tag::map[]
void eobj_build_map(int nw, const PaWrap *ww, /**/ EObjPack *p);
// end::map[]

// tag::pack[]
void eobj_pack(int nw, const PaWrap *ww, /**/ EObjPack *p);
void eobj_download(int nw, EObjPack *p);
// end::pack[]

// tag::com[]
void eobj_post_recv(EObjComm *c, EObjUnpack *u);
void eobj_post_send(EObjPack *p, EObjComm *c);
void eobj_wait_recv(EObjComm *c, EObjUnpack *u);
void eobj_wait_send(EObjComm *c);
// end::com[]

// tag::get[]
int26 eobj_get_counts(EObjUnpack *u); // <1>
Pap26 eobj_upload_shift(EObjUnpack *u); // <2>
Fop26 eobj_reini_ff(const EObjUnpack *u, EObjPackF *pf); // <3>
// end::get[]


// tag::memback[]
void eobj_pack_fin(EObjPack *p);
void eobj_comm_fin(EObjComm *c);
void eobj_unpack_fin(EObjUnpack *u);
void eobj_packf_fin(EObjPackF *p);
void eobj_unpackf_fin(EObjUnpackF *u);
// end::memback[]

// tag::packback[]
void eobj_download_ff(EObjPackF *p);
// end::packback[]

// tag::comback[]
void eobj_post_recv_ff(EObjComm *c, EObjUnpackF *u);
void eobj_post_send_ff(EObjPackF *p, EObjComm *c);
void eobj_wait_recv_ff(EObjComm *c, EObjUnpackF *u);
void eobj_wait_send_ff(EObjComm *c);
// end::comback[]

// tag::unpackback[]
void eobj_unpack_ff(const EObjUnpackF *u, const EObjPack *p, int nw, /**/ FoWrap *ww);
// end::unpackback[]
