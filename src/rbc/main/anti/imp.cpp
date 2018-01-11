#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <vector_types.h>

#include "utils/msg.h"
#include "utils/error.h"
#include "utils/imp.h"

#include "rbc/adj/type/common.h"
#include "rbc/adj/type/hst.h"
#include "rbc/adj/imp.h"
#include "rbc/edg/imp.h"

#include "imp.h"

namespace anti {
static void set(int md, int nv, adj::Adj *adj, /**/ int *hx, int *hy) {
    adj::AdjMap m;
    int valid, i, i0, i1;
    for (i = 0; i < md*nv; i++) {
        valid = adj::hst(md, nv, i, adj, /**/ &m);
        if (!valid) continue;
        i0 = m.i0; i1 = m.i1;
        edg_set(md, i0, i1, i,  hx, hy);
    }
}

static void get(int md, int nv, adj::Adj *adj, int *hx, int *hy, /**/ int *anti) {
    adj::AdjMap m;
    int valid, i, j, i0, i1;
    for (i = 0; i < md*nv; i++) {
        valid = adj::hst(md, nv, i, adj, /**/ &m);
        if (!valid) continue;
        i0 = m.i0; i1 = m.i1;
        /* invert i1 and i0 */
        j = edg_get(md, i1, i0, hx, hy);
        anti[i] = j;
    }
}

static void ini0(int md, int nv, adj::Adj *adj, /**/ int *anti, /*w*/ int *hx, int *hy) {
    edg_ini(md, nv, /**/ hx);
    set(md, nv, adj, /**/ hx, hy);
    get(md, nv, adj, hx, hy, /**/ anti);
}

void ini(int md, int nv, adj::Adj *adj, /**/ int *anti) {
    int n;
    int *hx, *hy;
    n = md*nv;
    UC(emalloc(n*sizeof(int), (void**) &hx));
    UC(emalloc(n*sizeof(int), (void**) &hy));
    ini0(md, nv, adj, /**/ anti, /*w*/ hx, hy);
    free(hx); free(hy);
}

} /* namespace */

