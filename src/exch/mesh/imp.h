namespace exch {
namespace mesh {

/* mesh exchanger */

void emesh_pack_ini(int nv, int max_mesh_num, Pack *p);
void emesh_comm_ini(MPI_Comm comm, /**/ Comm *c);
void emesh_unpack_ini(int nv, int max_mesh_num, Unpack *u);

void emesh_pack_fin(Pack *p);
void emesh_comm_fin(Comm *c);
void emesh_unpack_fin(Unpack *u);

void emesh_build_map(int nm, int nv, const Particle *pp, /**/ Pack *p);
void emesh_pack(int nv, const Particle *pp, /**/ Pack *p);
void emesh_download(Pack *p);

void emesh_post_recv(Comm *c, Unpack *u);
void emesh_post_send(Pack *p, Comm *c);
void emesh_wait_recv(Comm *c, Unpack *u);
void emesh_wait_send(Comm *c);

void emesh_unpack(int nv, const Unpack *u, /**/ int *nmhalo, Particle *pp);


/* optional: (back) momentum exchanger */

void emesh_get_num_frag_mesh(const Unpack *u, /**/ int cc[NFRAGS]);

void emesh_packm_ini(int num_mom_per_mesh, int max_mesh_num, PackM *p);
void emesh_commm_ini(MPI_Comm comm, /**/ CommM *c);
void emesh_unpackm_ini(int num_mom_per_mesh, int max_mesh_num, UnpackM *u);

void emesh_packm_fin(PackM *p);
void emesh_commm_fin(CommM *c);
void emesh_unpackm_fin(UnpackM *u);

void emesh_packM(int nt, const int counts[NFRAGS], const Momentum *mm, /**/ PackM *p);
void emesh_downloadM(const int counts[NFRAGS], PackM *p);

void emesh_post_recv(CommM *c, UnpackM *u);
void emesh_post_send(PackM *p, CommM *c);
void emesh_wait_recv(CommM *c, UnpackM *u);
void emesh_wait_send(CommM *c);

void emesh_upload(UnpackM *u);
void emesh_unpack_mom(int nt, const Pack *p, const UnpackM *u, /**/ Momentum *mm);

} // mesh
} // exch
