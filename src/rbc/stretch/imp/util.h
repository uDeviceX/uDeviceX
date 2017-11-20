static FILE* efopen(const char *path, const char *mode) {
    FILE *f;
    f = fopen(path, mode);
    if (f == NULL) ERR("fail to open <%s>", path);
    return f;
}
