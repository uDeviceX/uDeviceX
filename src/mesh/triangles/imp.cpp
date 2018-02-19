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
    int nv, nt;
    Triangles *q;
    const int4 *hst;
    int4 *dev;
    EMALLOC(1, &q);
    *pq = q;
    nv = mesh_get_nv(mesh);
    nt = mesh_get_nt(mesh);
    hst = mesh_get_tri(mesh);
    Dalloc(&dev, nt);
    cH2D(dev, hst, nt);

    q->nv = nv; q->nt = nt; q->tt = dev;
}

void triangles_fin(Triangles *q) {
    Dfree(q->tt);
    EFREE(q);
}
