namespace exch {
namespace mesh {

void ini(int nv, int max_mesh_num, Pack *p);
void ini(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ Comm *c);
void ini(int nv, int max_mesh_num, Unpack *u);

void fin(Pack *p);
void fin(Comm *c);
void fin(Unpack *u);


} // mesh
} // exch
