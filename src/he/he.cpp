#include <stdio.h>

#include "err.h"
#include "memory.h"
#include "read.h"

#include "he.h"

#define T He

#define MAGIC (43)

struct T {
    int nv, nt, ne, nh;
    int *nxt, *flp;
    int *ver, *tri, *edg;
    int *hdg_ver, *hdg_edg, *hdg_tri;
    int magic;
};

int he_file_ini(const char *path, T **pq) {
    HeRead *read;
    if (he_read_ini(path, &read) != HE_OK)
        ERR(HE_IO, "he_read_ini failed");
    he_ini(read, /**/ pq);
    he_read_fin(read);
    return HE_OK;
}

int he_ini(HeRead *r, T **pq) {
    T *q;
    int nv, nt, ne, nh;
    int *nxt, *flp, *ver, *tri, *edg;
    int *hdg_ver, *hdg_edg, *hdg_tri;
    MALLOC(1, &q);

    nv = q->nv = he_read_nv(r);
    nt = q->nt = he_read_nt(r);
    ne = q->ne = he_read_ne(r);
    nh = q->nh = he_read_nh(r);    

    MALLOC(nh, &q->nxt); MALLOC(nh, &q->flp);
    MALLOC(nh, &q->ver); MALLOC(nh, &q->tri); MALLOC(nh, &q->edg);
    MALLOC(nv, &q->hdg_ver);
    MALLOC(ne, &q->hdg_edg);
    MALLOC(nt, &q->hdg_tri);

    he_read_nxt(r, &nxt);
    he_read_flp(r, &flp);
    he_read_ver(r, &ver);
    he_read_tri(r, &tri);
    he_read_edg(r, &edg);

    he_read_hdg_ver(r, &hdg_ver);
    he_read_hdg_edg(r, &hdg_edg);
    he_read_hdg_tri(r, &hdg_tri);

    MEMCPY(nh, nxt, q->nxt); MEMCPY(nh, flp, q->flp);
    MEMCPY(nh, ver, q->ver); MEMCPY(nh, tri, q->tri); MEMCPY(nh, edg, q->edg);
    MEMCPY(nv, hdg_ver, q->hdg_ver);
    MEMCPY(ne, hdg_edg, q->hdg_edg);
    MEMCPY(nt, hdg_tri, q->hdg_tri);

    q->magic = MAGIC;
    
    *pq = q;
    return HE_OK;    
}

int he_fin(T *q) {
    if (q->magic != MAGIC)
        ERR(HE_MEMORY, "wrong fin() call");
    FREE(q->nxt); FREE(q->flp);
    FREE(q->ver); FREE(q->tri); FREE(q->edg);
    FREE(q->hdg_ver);
    FREE(q->hdg_edg);
    FREE(q->hdg_tri);
    FREE(q);
    return HE_OK;
}

int he_fin(T*);

int he_nv(T *q) { return q->nv; }
int he_nt(T *q) { return q->nt; }
int he_ne(T *q) { return q->ne; }
int he_nh(T *q) { return q->nh; }

/* validate */
#define V(i, n) if (0 > (i) || (i) >= n)                        \
        ERR(HE_INDEX, "%s=%d is not in [0, %d)", #i, i, n)
int he_nxt(T *q, int h) { V(h,q->nh); return q->nxt[h]; }
int he_flp(T *q, int h) {
    int f;
    V(h, q->nh);
    if ((f = q->flp[h]) == -1)
        ERR(HE_INDEX, "no flip for %d", h);
    return f;
}
int he_ver(T *q, int h) { V(h, q->nh); return q->ver[h]; }
int he_tri(T *q, int h) { V(h, q->nh); return q->tri[h]; }
int he_edg(T *q, int h) { V(h, q->nh); return q->edg[h]; }
int he_hdg_ver(T *q, int v) { V(v, q->nv); return q->hdg_ver[v]; }
int he_hdg_edg(T *q, int e) { V(e, q->ne); return q->hdg_edg[e]; }
int he_hdg_tri(T *q, int t) { V(t, q->nt); return q->hdg_tri[t]; }
int he_bnd(T *q, int h)     { V(h, q->nh); return q->flp[h] == -1; }
