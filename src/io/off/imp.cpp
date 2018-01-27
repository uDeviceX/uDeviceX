#include <vector_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils/error.h"
#include "utils/imp.h"

#include "utils/msg.h"

#include "imp.h"

#include "imp/type.h"
#include "imp/main.h"

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
