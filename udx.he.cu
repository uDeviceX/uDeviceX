// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/he/err.cpp */
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>

/* begin src/he/def.h */
enum {MAX_STRING_SIZE = 4096};
/* end src/he/def.h */
/* begin src/he/macro.h */
#ifdef __GNUC__
  #define __UNUSED __attribute__((unused))
#else
  #define __UNUSED
#endif
/* end src/he/macro.h */
/* begin src/he/err.h */
/* error codes */
enum {HE_OK, HE_IO, HE_MEMORY, HE_SYS, HE_USER, HE_INDEX, HE_NUM};

/* for user code */
#define ER(fmt, ...) ERR(HE_USER, (fmt), ##__VA_ARGS__)
#define ERR(code, fmt, ...)                                             \
    do {                                                                \
        he_err(code, __FILE__, __LINE__, (fmt), ##__VA_ARGS__);         \
        return code;                                                    \
    } while(0)

#define MSG(fmt, ...) he_msg(__FILE__, __LINE__, (fmt), ##__VA_ARGS__)

void he_err(int code, const char *file, int line, const char *fmt, ...);
void he_msg(const char *file, int line, const char *fmt, ...);
/* end src/he/err.h */

#define SIZE (MAX_STRING_SIZE)

void he_err(__UNUSED int code, const char *file, int line, const char *fmt, ...) {
    char msg[SIZE];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, SIZE, fmt, ap);
    va_end(ap);
    fprintf(stderr, "%s:%d: %s\n", file, line, msg);
    abort();
}

void he_msg(const char *file, int line, const char *fmt, ...) {
    char msg[SIZE];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, SIZE, fmt, ap);
    va_end(ap);
    fprintf(stderr, "%s:%d: %s\n", file, line, msg);
}
/* end src/he/err.cpp */

#line 1 "src/he/hash.cpp"


/* begin src/he/hash.cpp */
#include <stdio.h>
/* begin src/he/memory.h */
#define MALLOC(n, pp) he_malloc((n)*sizeof(**(pp)), (void**)(pp))
#define FREE(p) he_free(p)
#define MEMCPY(n, src, dest) he_memcpy((void*)(dest), (const void*)(src), (n)*sizeof(*(dest)))

int he_malloc(int, void**);
int he_free(void *ptr);
int he_memcpy(void *dest, const void *src, size_t n);
/* end src/he/memory.h */

/* begin src/he/hash.h */
#define T HeHash

typedef struct T T;

int he_hash_ini(int n, T**);

/* hash[i, j] = v */
int he_hash_set(T*, int i, int j, int v);

/* ans = hash[i, j], ans = -1 if unset  */
int he_hash_get(T*, int i, int j);
int he_hash_fin(T*);

#undef T
/* end src/he/hash.h */

#define T HeHash
#define E(fmt, ...) ERR(HE_INDEX, fmt, ##__VA_ARGS__);

typedef struct Node Node;
struct Node {
    int j;
    int v; /* value */
    Node  *nxt;
};
struct T {
    int n;
    Node **node;
};

int he_hash_ini(int n, T **pq) {
    T *q;
    int i;
    MALLOC(1, &q);
    MALLOC(q->n = n, &q->node);
    for (i = 0; i < n; i++)
        q->node[i] = NULL;
    *pq = q;
    return HE_OK;
}

static void free_node(Node *p) {
    Node *n;
    while (p != NULL) {
        n = p->nxt;
        FREE(p);
        p = n;
    }
}
int he_hash_fin(T *q) {
    int n, i;
    n = q->n;
    for (i = 0; i < n; i++)
        free_node(q->node[i]);
    FREE(q->node);
    FREE(q);
    return HE_OK;
}

int he_hash_set(T *q, int i, int j, int v) {
    int n;
    Node *nxt, *prv, *nnew;
    n = q->n;
    if (0 > i || i >= n || 0 > j || j >= n)
        E("[%d %d], n = %d", i, j, n);

    prv = q->node[i];
    while (prv != NULL) {
        if (prv->j == j) {
            prv->v = v;
            return HE_OK;
        }
        nxt = prv->nxt;
        if (nxt == NULL) break;
        prv = nxt;
    }

    MALLOC(1, &nnew);
    nnew->j = j;
    nnew->v = v;
    nnew->nxt = NULL;
    if (prv == NULL) q->node[i] = nnew;
    else prv->nxt = nnew;
    return HE_OK;
}

int he_hash_get(T *q, int i, int j) {
    int n;
    Node *node;
    n = q->n;
    if (0 > i || i >= n || 0 > j || j >= n)
        E("[%d %d], n = %d", i, j, n);
    node = q->node[i];
    while (node != NULL) {
        if (node->j == j) return node->v;
        node = node->nxt;
    }
    return -1;
}
/* end src/he/hash.cpp */

#line 1 "src/he/he.cpp"


/* begin src/he/he.cpp */
#include <stdio.h>



/* begin src/he/read.h */
#define T HeRead

struct int4;

typedef struct T T;

int he_read_ini(const char *path, T**);

/* tri = [t0, t1, t2] [t0, t2, t2], ... */
int he_read_tri_ini(int nv, int nt, int *tri, T**);

/* tri = [tri[0].x tri[0].y tri[0].z], ... [tri[0].x tri[0].y tri[0].z] */
int he_read_int4_ini(int nv, int nt, const int4 *tri, T**);

int he_read_fin(T*);

int he_read_nv(T*);
int he_read_nt(T*);
int he_read_ne(T*);
int he_read_nh(T*);

int he_read_nxt(T*, int**);
int he_read_flp(T*, int**);
int he_read_ver(T*, int**);
int he_read_tri(T*, int**);
int he_read_edg(T*, int**);

int he_read_hdg_ver(T*, int**);
int he_read_hdg_edg(T*, int**);
int he_read_hdg_tri(T*, int**);

int he_read_info(T*, FILE*);

#undef T
/* end src/he/read.h */

/* begin src/he/he.h */
#define T He

typedef struct T T;
typedef struct HeRead HeRead;

int he_ini(HeRead*, T**);
int he_file_ini(const char *path, T**);
int he_fin(T*);

int he_nv(T*); /* number of vert, edg, edg, hdg */
int he_nt(T*);
int he_ne(T*);
int he_nh(T*);

int he_nxt(T*, int h); /* return half-edg */
int he_flp(T*, int h);

int he_ver(T*, int h); /* return ver, tri, or edg */
int he_tri(T*, int h);
int he_edg(T*, int h);

int he_hdg_ver(T*, int v); /* return half-edg */
int he_hdg_edg(T*, int e);
int he_hdg_tri(T*, int t);

int he_bnd(T*, int h); /* 1 for boundary */

#undef T
/* end src/he/he.h */

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
/* end src/he/he.cpp */

#line 1 "src/he/memory.cpp"


/* begin src/he/memory.cpp */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>




int he_malloc(int size, /**/ void **pq) {
    void *q;
    q = malloc(size);
    if (q == NULL) ERR(HE_MEMORY, "malloc failed for size=%ld", size);
    *pq = q;
    return HE_OK;
}

int he_free(void *q) {
    free(q);
    q = NULL;
    return HE_OK;
}

int he_memcpy(void *dest, const void *src, size_t n) {
    if (dest == NULL) ERR(HE_MEMORY, "dest == NULL");
    if (src == NULL) ERR(HE_MEMORY, "src == NULL");
    memcpy(dest, src, n);
    return HE_OK;
}
/* end src/he/memory.cpp */

#line 1 "src/he/read.cpp"


/* begin src/he/read.cpp */
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <vector_types.h>





/* begin src/he/util.h */
/* string equal? */
int util_eq(const char *a, const char *b);

/* fgets without '\n' */
char *util_fgets(char *s, FILE *stream);

/* fgets without '\n' and skipping '#comments' */
char *util_comment_fgets(char *s, FILE *stream);
/* end src/he/util.h */




#define T HeRead
#define SIZE (MAX_STRING_SIZE)
#define E(fmt, ...) ERR(HE_IO, fmt, ##__VA_ARGS__);
#define MAGIC (42)

struct T {
    int nv, nt, ne, nh;
    int *nxt, *flp;
    int *ver, *tri, *edg;
    int *hdg_ver, *hdg_edg, *hdg_tri;
    int magic;
};


enum {OK, FAIL};
static int valid(int n, int *a, int *b) {
    int i, i0;
    for (i = 0; i < n; i++) {
        i0 = b[a[i]];
        if (i != i0) {
            MSG("i=%d a[i]=%d i0=%d ", i, a[i], i0);
            return FAIL;
        }
    }
    return OK;
}

static int alloc_hdg(int nh, T *q) {
    MALLOC(nh, &q->nxt); MALLOC(nh, &q->flp);
    MALLOC(nh, &q->ver); MALLOC(nh, &q->tri); MALLOC(nh, &q->edg);
    return HE_OK;
}
static int alloc_ver(int nv, T *q) { MALLOC(nv, &q->hdg_ver); return HE_OK; }
static int alloc_edg(int ne, T *q) { MALLOC(ne, &q->hdg_edg); return HE_OK; }
static int alloc_tri(int nt, T *q) { MALLOC(nt, &q->hdg_tri); return HE_OK; }
static int alloc(int nv, int ne, int nt, int nh, T *q) {
    alloc_hdg(nh, q);
    alloc_ver(nv, q); alloc_edg(ne, q); alloc_tri(nt, q);
    return HE_OK;
}

static int afree(T *q) {
    FREE(q->nxt); FREE(q->flp);
    FREE(q->ver); FREE(q->tri); FREE(q->edg);
    FREE(q->hdg_ver);
    FREE(q->hdg_edg);
    FREE(q->hdg_tri);
    return HE_OK;
}

static void setup_edg(T *q, HeHash *hdg, int i, int j,  int *pe) {
    int h, f, e;
    h = he_hash_get(hdg, i, j);
    if (h == -1) return;
    e = *pe;

    q->edg[h] = e;
    if ((f = he_hash_get(hdg, j, i)) != -1)
        q->edg[f] = e;

    he_hash_set(hdg, i, j, -1); /* counted it */
    he_hash_set(hdg, j, i, -1);

    e++;
    *pe = e;
}
static void setup_hdg(T *q, HeHash *hdg, int h, int t, int i, int j) {
    he_hash_set(hdg, i, j, h);
    q->hdg_tri[t] = q->hdg_ver[i] = h;
    q->ver[h] = i; q->tri[h] = t;
}
static void setup_flip(T *q, HeHash *hdg, int i, int j) {
    int h, f;
    h = he_hash_get(hdg, i, j);
    f = he_hash_get(hdg, j, i);
    q->flp[h] = f;
}

int he_read_int4_ini(int nv, int nt, const int4 *tt, T **pq) {
    int status;
    int *tri;
    int i, j;
    MALLOC(3*nt, &tri);
    for (j = i = 0; i < nt; i++) {
        tri[j++] = tt[i].x;
        tri[j++] = tt[i].y;
        tri[j++] = tt[i].z;
    }
    status = he_read_tri_ini(nv, nt, tri, pq);
    FREE(tri);
    return status;
}

int he_read_tri_ini(int nv, int nt, int *tri0, T **pq) {
    int ne, nh;
    int e, t, i, j, k;
    int h, hi, hj, hk;
    int *tri;
    T *q;
    HeHash *hdg; /* will maps [i, j] to half-edg */
    MALLOC(1, &q);

    alloc_ver(nv,        q);
    alloc_tri(nt,        q);
    alloc_hdg(nh = 3*nt, q);
    he_hash_ini(nv, &hdg);
    for (tri = tri0, t = h = 0; t < nt; t++) {
        i = *tri++; j = *tri++; k = *tri++;
        assert(i < nv); assert(j < nv); assert(k < nv);
        setup_hdg(q, hdg, hi = h++, t, i, j);
        setup_hdg(q, hdg, hj = h++, t, j, k);
        setup_hdg(q, hdg, hk = h++, t, k, i);
        q->nxt[hi] = hj; q->nxt[hj] = hk; q->nxt[hk] = hi;
    }

    for (tri = tri0, t = h = 0; t < nt; t++) {
        i = *tri++; j = *tri++; k = *tri++;
        setup_flip(q, hdg, i, j);
        setup_flip(q, hdg, j, k);
        setup_flip(q, hdg, k, i);
    }

    for (tri = tri0, ne = t = h = 0; t < nt; t++) {
        i = *tri++; j = *tri++; k = *tri++;
        setup_edg(q, hdg, i, j,  &ne);
        setup_edg(q, hdg, j, k,  &ne);
        setup_edg(q, hdg, k, i,  &ne);
    }
    he_hash_fin(hdg);

    alloc_edg(ne, q);
    for (h = 0; h < nh; h++) {
        e = q->edg[h];
        q->hdg_edg[e] = h;
    }

    q->nv = nv; q->ne = ne; q->nt = nt; q->nh = nh;
    q->magic = MAGIC;
    *pq = q;
    return HE_OK;
}

int he_read_ini(const char *path, T **pq) {
    T *q;
    FILE *f;
    char line[SIZE];
    int i, cnt;
    int nv, nt, ne, nh;

#   define NXT() if (util_fgets(line, f) == NULL)  \
        E("unexpected EOF in '%s'", path)
    MALLOC(1, &q);
    f = fopen(path, "r");
    if (f == NULL) E("fail to open '%s'", path);
    NXT();
    if (!util_eq(line, "HE"))
        E("'%s' is not a he file", path);
    NXT();
    cnt = sscanf(line, "%d %d %d %d", &nv, &ne, &nt, &nh);
    if (cnt != 4)
        E("'%s' != [nv nt ne nh] in '%s'", line, path);
    if (nv <= 0 || nt <= 0 || ne <= 0 || nh <= 0)
        E("wrong sizes '%s' in '%s'", line, path);

    alloc(nv, ne, nt, nh, /**/ q);

    for (i = 0; i < nh; i++) {
        NXT();
        cnt = sscanf(line, "%d %d  %d %d %d",
                     &q->nxt[i], &q->flp[i], &q->ver[i], &q->edg[i], &q->tri[i]);
        if (cnt != 5) E("wrong half-edg line '%s' in '%s'", line, path);
    }
    for (i = 0; i < nv; i++) {
        NXT();
        cnt = sscanf(line, "%d", &q->hdg_ver[i]);
        if (cnt != 1) E("wrong ver line '%s' in '%s'", line, path);
    }
    for (i = 0; i < ne; i++) {
        NXT();
        cnt = sscanf(line, "%d", &q->hdg_edg[i]);
        if (cnt != 1) E("wrong edg line '%s' in '%s'", line, path);
    }
    for (i = 0; i < nt; i++) {
        NXT();
        cnt = sscanf(line, "%d", &q->hdg_tri[i]);
        if (cnt != 1) E("wrong tri line '%s' in '%s'", line, path);
    }
    if (fclose(f) != 0)
        E("fail to close file '%s'", path);

    q->nv = nv; q->nt = nt; q->ne = ne; q->nh = nh;
    q->magic = MAGIC;

    if (valid(nv, q->hdg_ver, q->ver) != OK)
        E("invalid ver references");
    if (valid(ne, q->hdg_edg, q->edg) != OK)
        E("invalid edg references");
    if (valid(nt, q->hdg_tri, q->tri) != OK)
        E("invalid tri references");

    *pq = q;
    return HE_OK;
}

int he_read_fin(T *q) {
    if (q->magic != MAGIC)
        ERR(HE_MEMORY, "wrong fin() call");
    afree(q);
    FREE(q);
    return HE_OK;
}

int he_read_info(T *q, FILE *f) {
    int r;
    int nv, nt, ne, nh;
    int *nxt, *flp, *ver, *tri, *edg;
    int *hdg_ver, *hdg_edg, *hdg_tri;

    nv = q->nv; nt = q->nt; ne = q->ne; nh = q->nh;
    nxt = q->nxt; flp = q->flp;
    ver = q->ver; tri = q->tri; edg = q->edg;
    hdg_ver = q->hdg_ver;
    hdg_edg = q->hdg_edg;
    hdg_tri = q->hdg_tri;

    r = fprintf(f, "%d %d %d %d\n", nv, ne, nt, nh);
    if (r <= 0)
        ERR(HE_IO, "fprintf() failed");

    fprintf(f, "[nh=%d lines]\n", nh);
    fprintf(f, "%d %d %d %d %d\n",
            nxt[0], flp[0], ver[0], tri[0], edg[0]);
    fputs("...\n", f);
    fprintf(f, "%d %d %d %d %d\n",
            nxt[nh-1], flp[nh-1], ver[nh-1], tri[nh-1], edg[nh-1]);
    fprintf(f, "[nv=%d lines]\n", nv);
    fprintf(f, "%d\n", hdg_ver[0]);
    fputs("...\n", f);
    fprintf(f, "%d\n", hdg_ver[nv-1]);

    fprintf(f, "[ne=%d lines]\n", ne);
    fprintf(f, "%d\n", hdg_edg[0]);
    fputs("...\n", f);
    fprintf(f, "%d\n", hdg_edg[ne-1]);

    fprintf(f, "[nt=%d lines]\n", nt);
    fprintf(f, "%d\n", hdg_tri[0]);
    fputs("...\n", f);
    fprintf(f, "%d\n", hdg_tri[nt-1]);

    return HE_OK;
}

int he_read_nv(T *q) { return q->nv; }
int he_read_nt(T *q) { return q->nt; }
int he_read_ne(T *q) { return q->ne; }
int he_read_nh(T *q) { return q->nh; }

int he_read_nxt(T *q, int **p) { *p = q->nxt; return HE_OK; };
int he_read_flp(T *q, int **p) { *p = q->flp; return HE_OK; };
int he_read_ver(T *q,  int **p) { *p = q->ver; return HE_OK; };
int he_read_tri(T *q,  int **p) { *p = q->tri; return HE_OK; };
int he_read_edg(T *q,  int **p) { *p = q->edg; return HE_OK; };

int he_read_hdg_ver(T *q, int **p) { *p = q->hdg_ver; return HE_OK; };
int he_read_hdg_edg(T *q, int **p) { *p = q->hdg_edg; return HE_OK; };
int he_read_hdg_tri(T *q, int **p) { *p = q->hdg_tri; return HE_OK; };
/* end src/he/read.cpp */

#line 1 "src/he/util.cpp"


/* begin src/he/util.cpp */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>






#define  SIZE (MAX_STRING_SIZE)

int util_eq(const char *a, const char *b) {
    return strncmp(a, b, SIZE) == 0;
}

char *util_fgets(char *s, FILE *stream) {
    char *c;
    if (fgets(s, SIZE, stream) == NULL)
        return NULL;
    if ((c = strchr(s, '\n')) != NULL)
        *c = '\0';
    return s;
}

enum {NO, YES};
static int commentp(const char *s) {
    while (*(s++) != '\0') {
        if    (*s == '#') return YES;
        if (!isblank(*s)) return NO;
    }
    return YES;
}
char *util_comment_fgets(char *s, FILE *stream) {
    do if (util_fgets(s, stream) == NULL)
           return NULL;
    while (commentp(s) == YES);
    return s;
}
/* end src/he/util.cpp */

#line 1 "src/inter/color/conf.cpp"
