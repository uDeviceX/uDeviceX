static void *emalloc(size_t size) {
    void *p;
    p = malloc(size);
    if (p == NULL) ERR("out of memory: requested: %ld", size);
    return p;
}

static FILE* efopen(const char *path, const char *mode) {
    FILE *f;
    f = fopen(path, mode);
    if (f == NULL) ERR("fail to open: %s\n", path);
    return f;
}
