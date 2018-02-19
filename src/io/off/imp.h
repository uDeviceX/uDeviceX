struct int4;
struct MeshRead;

/* off files
   [1] https://en.wikipedia.org/wiki/OFF_(file_format) */

void mesh_read_off(const char *path, MeshRead**);
void mesh_read_ply(const char *path, MeshRead**);
void mesh_fin(MeshRead*);

int mesh_get_nt(MeshRead*);
int mesh_get_nv(MeshRead*);
int mesh_get_md(MeshRead*); /* maximum vertex degree */

const int4  *mesh_get_tri(MeshRead*);
const float *mesh_get_vert(MeshRead*);
