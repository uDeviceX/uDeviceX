namespace off {
/* file to vertices : max: maximum vert number */
void vert(const char *f, int max, /**/ int *nv, float *vert);

/* file to faces : max: maximum face number */
void faces(const char *f, int max, /**/ int *nf, int4 *faces);
}
