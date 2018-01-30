struct EMeshPack;
struct EMeshComm;
struct EMeshUnpack;
struct EMeshPackM;
struct EMeshCommM;
struct EMeshUnpackM;

/* mesh exchanger */

void emesh_pack_ini(int3 L, int nv, int max_mesh_num, EMeshPack **p);
void emesh_comm_ini(MPI_Comm comm, /**/ EMeshComm **c);
void emesh_unpack_ini(int3 L, int nv, int max_mesh_num, EMeshUnpack **u);

void emesh_pack_fin(EMeshPack *p);
void emesh_comm_fin(EMeshComm *c);
void emesh_unpack_fin(EMeshUnpack *u);

void emesh_build_map(int nm, int nv, const Particle *pp, /**/ EMeshPack *p);
void emesh_pack(int nv, const Particle *pp, /**/ EMeshPack *p);
void emesh_download(EMeshPack *p);

void emesh_post_recv(EMeshComm *c, EMeshUnpack *u);
void emesh_post_send(EMeshPack *p, EMeshComm *c);
void emesh_wait_recv(EMeshComm *c, EMeshUnpack *u);
void emesh_wait_send(EMeshComm *c);

void emesh_unpack(int nv, const EMeshUnpack *u, /**/ int *nmhalo, Particle *pp);


/* optional: (back) momentum exchanger */

void emesh_get_num_frag_mesh(const EMeshUnpack *u, /**/ int cc[NFRAGS]);

void emesh_packm_ini(int num_mom_per_mesh, int max_mesh_num, EMeshPackM **p);
void emesh_commm_ini(MPI_Comm comm, /**/ EMeshCommM **c);
void emesh_unpackm_ini(int num_mom_per_mesh, int max_mesh_num, EMeshUnpackM **u);

void emesh_packm_fin(EMeshPackM *p);
void emesh_commm_fin(EMeshCommM *c);
void emesh_unpackm_fin(EMeshUnpackM *u);

void emesh_packM(int nt, const int counts[NFRAGS], const Momentum *mm, /**/ EMeshPackM *p);
void emesh_downloadM(const int counts[NFRAGS], EMeshPackM *p);

void emesh_post_recv(EMeshCommM *c, EMeshUnpackM *u);
void emesh_post_send(EMeshPackM *p, EMeshCommM *c);
void emesh_wait_recv(EMeshCommM *c, EMeshUnpackM *u);
void emesh_wait_send(EMeshCommM *c);

void emesh_upload(EMeshUnpackM *u);
void emesh_unpack_mom(int nt, const EMeshPack *p, const EMeshUnpackM *u, /**/ Momentum *mm);
