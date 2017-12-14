int emalloc(size_t, /**/ void **);
void efree(void*);

int efopen(const char *fname, const char *mode, /**/ FILE **f);
void efclose(FILE *f);
