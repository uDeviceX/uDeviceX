struct int4;
struct MeshRead;

// tag::interface[]
void mesh_read_off(const char *path, MeshRead**);
void mesh_read_ply(const char *path, MeshRead**);
void mesh_fin(MeshRead*);

int mesh_get_nt(MeshRead*);
int mesh_get_nv(MeshRead*);
int mesh_get_md(MeshRead*);

const int4  *mesh_get_tri(MeshRead*);
const float *mesh_get_vert(MeshRead*);
// end::interface[]
