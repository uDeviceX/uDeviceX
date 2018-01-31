void emalloc(size_t, /**/ void**);
void efree(void*);

#define EMALLOC(n, ppdata) UC(emalloc(n * sizeof(**(ppdata)), (void**) (ppdata)))
#define EFREE(pdata) UC(efree(pdata))

void efopen(const char *fname, const char *mode, /**/ FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b);
