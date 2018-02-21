struct EMeshPack;
struct EMeshComm;
struct EMeshUnpack;
struct EMeshPackM;
struct EMeshCommM;
struct EMeshUnpackM;

/* mesh exchanger */

// tag::mem[]
void emesh_pack_ini(int3 L, int nv, int max_mesh_num, EMeshPack **p);
void emesh_comm_ini(MPI_Comm comm, /**/ EMeshComm **c);
void emesh_unpack_ini(int3 L, int nv, int max_mesh_num, EMeshUnpack **u);

void emesh_pack_fin(EMeshPack *p);
void emesh_comm_fin(EMeshComm *c);
void emesh_unpack_fin(EMeshUnpack *u);
// end::mem[]

// tag::map[]
void emesh_build_map(int nm, int nv, const Particle *pp, /**/ EMeshPack *p);
// end::map[]

// tag::pack[]
void emesh_pack(int nv, const Particle *pp, /**/ EMeshPack *p);
void emesh_download(EMeshPack *p);
// end::pack[]

// tag::com[]
void emesh_post_recv(EMeshComm *c, EMeshUnpack *u);
void emesh_post_send(EMeshPack *p, EMeshComm *c);
void emesh_wait_recv(EMeshComm *c, EMeshUnpack *u);
void emesh_wait_send(EMeshComm *c);
// end::com[]

// tag::unpack[]
void emesh_unpack(int nv, const EMeshUnpack *u, /**/ int *nmhalo, Particle *pp);
// end::unpack[]

// tag::get[]
void emesh_get_num_frag_mesh(const EMeshUnpack *u, /**/ int cc[NFRAGS]);
// end::get[]

/* optional: (back) momentum exchanger */

// tag::memback[]
void emesh_packm_ini(int num_mom_per_mesh, int max_mesh_num, EMeshPackM **p);
void emesh_commm_ini(MPI_Comm comm, /**/ EMeshCommM **c);
void emesh_unpackm_ini(int num_mom_per_mesh, int max_mesh_num, EMeshUnpackM **u);

void emesh_packm_fin(EMeshPackM *p);
void emesh_commm_fin(EMeshCommM *c);
void emesh_unpackm_fin(EMeshUnpackM *u);
// end::memback[]

// tag::packback[]
void emesh_packM(int nt, const int counts[NFRAGS], const Momentum *mm, /**/ EMeshPackM *p);
void emesh_downloadM(const int counts[NFRAGS], EMeshPackM *p);
// end::packback[]

// tag::comback[]
void emesh_post_recv(EMeshCommM *c, EMeshUnpackM *u);
void emesh_post_send(EMeshPackM *p, EMeshCommM *c);
void emesh_wait_recv(EMeshCommM *c, EMeshUnpackM *u);
void emesh_wait_send(EMeshCommM *c);
// end::comback[]

// tag::unpackback[]
void emesh_upload(EMeshUnpackM *u);
void emesh_unpack_mom(int nt, const EMeshPack *p, const EMeshUnpackM *u, /**/ Momentum *mm);
// end::unpackback[]
