struct int4;
struct MeshRead;

// tag::interface[]
void mesh_read_off(const char *path, MeshRead**);
void mesh_read_ply(const char *path, MeshRead**);
void mesh_fin(MeshRead*);

int mesh_get_nt(const MeshRead*);
int mesh_get_nv(const MeshRead*);
int mesh_get_md(const MeshRead*);

const int4  *mesh_get_tri(const MeshRead*);
const float *mesh_get_vert(const MeshRead*);
// end::interface[]
