struct Mesh;
struct MeshRead;

void mesh_ini(MeshRead*, /**/ Mesh**);
void mesh_copy(const Mesh*, /**/ Mesh**);
void mesh_fin(Mesh*);

int mesh_nv(Mesh*);
int mesh_nt(Mesh*);
int mesh_ne(Mesh*);
const int* mesh_tt(Mesh*);
