static void efaces(const char *f, int n0, /**/ int4 *faces) {
    /* get faces */
    int n;
    n = off::faces(f, n0, faces);
    if (n0 != n)
        ERR("wrong faces number in <%s> : %d != %d", f, n0, n);
}

static void evert(const char *f, int n0, /**/ float *vert) {
    /* get vertices */
    int n;
    n = off::vert(f, n0, vert);
    if (n0 != n)
        ERR("wrong vert number in <%s> : %d != %d", f, n0, n);
}
