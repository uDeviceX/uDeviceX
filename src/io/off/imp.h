/* off files
   [1] https://en.wikipedia.org/wiki/OFF_(file_format) */

/* file to vertices : max: maximum vert number */
void off_read_vert(const char *f, int max, /**/ int *nv, float *vert);

/* file to faces : max: maximum face number */
void off_read_faces(const char *f, int max, /**/ int *nf, int4 *faces);
