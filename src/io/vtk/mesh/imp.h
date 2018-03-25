struct Mesh;
struct MeshRead;

void mesh_ini(MeshRead*, /**/ Mesh**);
void mesh_copy(const Mesh*, /**/ Mesh**);
void mesh_fin(Mesh*);

int mesh_nv(Mesh*);
