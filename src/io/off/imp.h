/* off files
   [1] https://en.wikipedia.org/wiki/OFF_(file_format) */

static int eq(const char *a, const char *b) { return strcmp(a, b) == 0; }

/* return faces: f0[0] f1[0] f2[0]   f0[1] f1[1] ... */
int faces(const char *f, int4* faces) {
    char buf[BUFSIZ];
    FILE *fd = fopen(f, "r");
    if (fd == NULL) {
        fprintf(stderr, "off: Could not open <%s>", f);
        exit(2);
    }
    fgets(buf, sizeof buf, fd); /* skip OFF */
    if (!eq(buf, "OFF\n")) {
        fprintf(stderr, "off: expecting [OFF] <%s> : [%s]\n", f, buf);
        exit(2);
    }

    int nv, nf;
    fscanf(fd, "%d %d %*d", &nv, &nf); /* skip `ne' and all vertices */
    for (int iv = 0; iv < nv;  iv++) fscanf(fd, "%*e %*e %*e");

    int4 t;
    t.w = 0;
    for (int ifa = 0; ifa < nf; ifa++) {
        fscanf(fd, "%*d %d %d %d", &t.x, &t.y, &t.z);
        faces[ifa] = t;
    }
    fclose(fd);

    return nf;
}

int vert(const char *f, float* vert) {
    char buf[BUFSIZ];
    FILE *fd = fopen(f, "r");
    if (fd == NULL) {
        fprintf(stderr, "off: Could not open <%s>", f);
        exit(2);
    }
    fgets(buf, sizeof buf, fd); /* skip OFF */

    int nv;
    fscanf(fd, "%d %*d %*d", &nv); /* skip `nf' and `ne' */
    int iv = 0, ib = 0;
    for (/*   */ ; iv < nv;  iv++) {
        float x, y, z;
        fscanf(fd, "%e %e %e", &x, &y, &z);
        vert[ib++] = x; vert[ib++] = y; vert[ib++] = z;
    }

    fclose(fd);

    return nv;
}
