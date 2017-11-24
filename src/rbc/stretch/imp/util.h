static int efopen(const char *path, const char *mode, /**/ FILE **pf) {
    FILE *f;
    f = fopen(path, mode);
    if (f == NULL)
        ERR("fail to open <%s>", path);
    *pf = f;
    return 0;
}
