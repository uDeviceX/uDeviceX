struct int4;
struct OffRead;

/* off files
   [1] https://en.wikipedia.org/wiki/OFF_(file_format) */

void off_read_off(const char *path, OffRead**);
void off_read_ply(const char *path, OffRead**);
void off_fin(OffRead*);

int off_get_nt(OffRead*);
int off_get_nv(OffRead*);
int off_get_md(OffRead*); /* maximum vertex degree */

const int4  *off_get_tri(OffRead*);
const float *off_get_vert(OffRead*);
