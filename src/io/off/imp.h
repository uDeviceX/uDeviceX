struct int4;

/* off files
   [1] https://en.wikipedia.org/wiki/OFF_(file_format) */

/* file to vertices : max: maximum vert number */
void off_read_vert(const char *f, int max, /**/ int *nv, float *vert);

/* file to faces : max: maximum face number */
void off_read_faces(const char *f, int max, /**/ int *nf, int4 *faces);

/***/
struct OffRead;
void off_read(const char *path, OffRead**);
void off_fin(OffRead*);

int    off_get_n(OffRead*);
int4  *off_get_tri(OffRead*);
float *off_get_vert(OffRead*);
/***/
