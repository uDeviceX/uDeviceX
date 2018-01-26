#include <vector_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils/error.h"
#include "utils/imp.h"

#include "imp.h"

static int eq(const char *a, const char *b) { return strcmp(a, b) == 0; }
static void assert_nf(int n, int max, const char *f) {
    if (n <= max) return;
    ERR("faces nf = %d < max = %d in <%s>", n, max, f);
}
/* return faces: f0[0] f1[0] f2[0]   f0[1] f1[1] ... */
void off_read_faces(const char *f, int max, /**/ int *pnf, int4 *faces) {
    char buf[BUFSIZ];
    FILE *fd;
    int nv, nf;
    int4 t;

    UC(efopen(f, "r", /**/ &fd));
    UC(efgets(buf, sizeof(buf), fd));
    if (!eq(buf, "OFF\n"))
        ERR("expecting [OFF] <%s> : [%s]", f, buf);

    fscanf(fd, "%d %d %*d", &nv, &nf); /* skip `ne' and all vertices */
    assert_nf(nf, max, f);

    for (int iv = 0; iv < nv;  iv++) fscanf(fd, "%*e %*e %*e");

    t.w = 0;
    for (int ifa = 0; ifa < nf; ifa++) {
        fscanf(fd, "%*d %d %d %d", &t.x, &t.y, &t.z);
        faces[ifa] = t;
    }
    UC(efclose(fd));

    *pnf = nf;
}

static void assert_nv(int n, int max, const char *f) {
    if (n <= max) return;
    ERR("vert nv = %d < max = %d in <%s>", n, max, f);
}

void off_read_vert(const char *f, int max, /**/ int *pnv, float *vert) {
    char buf[BUFSIZ];
    FILE *fd;
    int nv;
    int iv = 0, ib = 0;
    float x, y, z;

    UC(efopen(f, "r", /**/ &fd));

    UC(efgets(buf, sizeof buf, fd));
    if (!eq(buf, "OFF\n"))
        ERR("expecting [OFF] <%s> : [%s]", f, buf);

    fscanf(fd, "%d %*d %*d", &nv); /* skip `nf' and `ne' */
    assert_nv(nv, max, f);

    for (/*   */ ; iv < nv;  iv++) {
        fscanf(fd, "%e %e %e", &x, &y, &z);
        vert[ib++] = x; vert[ib++] = y; vert[ib++] = z;
    }

    UC(efclose(fd));

    *pnv = nv;
}

struct OffRead {
    int n;
    int4 *tt; /* triangles */
    int  *rr;
};

static void ini(OffRead **pq) {
    OffRead *p;
    UC(emalloc(sizeof(OffRead), (void**)&p));
    p->n  = -1;
    *pq = p;
}

void off_read(const char *path, OffRead **pq) {
    FILE *f;
    OffRead *p;
    UC(ini(&p));
    UC(efopen(path, "r", /**/ &f));

    UC(efclose(f));
    *pq = p;
}

void off_fin(OffRead* q) {
    //    UC(efree(q->rr));
    //    UC(efree(q->tt));
    UC(efree(q));
}

int    off_get_n(OffRead*) {
    return 0;
}

int4  *off_get_tri(OffRead*) {
    int4 *q = NULL;
    return q;
}

float *off_get_vert(OffRead*) {
    float *q = NULL;
    return q;
}
