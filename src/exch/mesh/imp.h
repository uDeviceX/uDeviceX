namespace exch {
namespace mesh {

/* mesh exchanger */

void ini(int nv, int max_mesh_num, Pack *p);
void ini(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ Comm *c);
void ini(int nv, int max_mesh_num, Unpack *u);

void fin(Pack *p);
void fin(Comm *c);
void fin(Unpack *u);

void build_map(int nm, int nv, const Particle *pp, /**/ Pack *p);
void pack(int nv, const Particle *pp, /**/ Pack *p);
void download(Pack *p);

void post_recv(Comm *c, Unpack *u);
void post_send(Pack *p, Comm *c);
void wait_recv(Comm *c, Unpack *u);
void wait_send(Comm *c);

void unpack(int nv, const Unpack *u, /**/ int *nmhalo, Particle *pp);


/* optional: (back) momentum exchanger */

void ini(int num_mom_per_mesh, int max_mesh_num, PackM *p);
void ini(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ CommM *c);
void ini(int num_mom_per_mesh, int max_mesh_num, UnpackM *u);

void fin(PackM *p);
void fin(CommM *c);
void fin(UnpackM *u);

void packM(const int counts[], const Momentum *mm, /**/ PackM *p);
void downloadM(PackM *p);

void post_recv(CommM *c, UnpackM *u);
void post_send(PackM *p, CommM *c);
void wait_recv(CommM *c, UnpackM *u);
void wait_send(CommM *c);


} // mesh
} // exch
