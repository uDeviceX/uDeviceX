struct int4;
struct MeshRead;

/* off files
   [1] https://en.wikipedia.org/wiki/OFF_(file_format) */

void off_read_off(const char *path, MeshRead**);
void off_read_ply(const char *path, MeshRead**);
void off_fin(MeshRead*);

int off_get_nt(MeshRead*);
int off_get_nv(MeshRead*);
int off_get_md(MeshRead*); /* maximum vertex degree */

const int4  *off_get_tri(MeshRead*);
const float *off_get_vert(MeshRead*);
