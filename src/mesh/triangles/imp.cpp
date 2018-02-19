#include <stdio.h>
#include <vector_types.h>

#include <conf.h>
#include "inc/conf.h"

#include "d/api.h"
#include "inc/dev.h"
#include "utils/cc.h"
#include "utils/imp.h"
#include "utils/error.h"

#include "io/off/imp.h"

#include "type.h"
#include "imp.h"

void triangles_ini(MeshRead *mesh, /**/ Triangles **pq) {
    int nt;
    Triangles *q;
    const int4 *hst;
    int4 *dev;
    EMALLOC(1, &q);
    *pq = q;
    nt = mesh_get_nt(mesh);
    hst = mesh_get_tri(mesh);
    Dalloc(&dev, nt);
    cH2D(dev, hst, nt);

    q->nt = nt; q->tt = dev;
}

void triangles_fin(Triangles *q) {
    Dfree(q->tt);
    EFREE(q);
}
