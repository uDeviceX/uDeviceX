// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/rbc/adj/imp.cpp */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector_types.h>

/* begin src/inc/conf.h */
/* [k]ernel [l]aunch macros */
#if !defined(KL_RELEASE)    && !defined(KL_NONE)   &&   \
    !defined(KL_SYNC)
#define KL_RELEASE
#endif

/* [c]uda [c]heck macro */
#if !defined(CC_RELEASE) && !defined(CC_SYNC)
  #define CC_RELEASE
#endif

/* who plays as device? */
#if !defined(DEV_CUDA) && !defined(DEV_CPU)
  #define DEV_CUDA
#endif

/* cuda memory allocation calls */
#if !defined(CUDA_ALLOC_RELEASE) && !defined(CUDA_ALLOC_DEBUG)
  #define CUDA_ALLOC_RELEASE
#endif

/* compute rbc force in double or float */
#if !defined(RBC_DOUBLE) && !defined(RBC_FLOAT)
  #define RBC_DOUBLE
#endif

/* what to do if r > lmax for a spring? */
#if !defined(RBC_SPRING_IGNORE) && !defined(RBC_SPRING_FAIL)
  #define RBC_SPRING_IGNORE
#endif

#ifndef DUMP_BASE
#define DUMP_BASE "."
#endif
/* end src/inc/conf.h */


/* begin src/utils/error.h */
#define ERR(fmt, ...)  do {                                     \
        error_signal(__FILE__, __LINE__, (fmt), ##__VA_ARGS__); \
        error_report();                                         \
        error_abort();                                          \
    } while (0)

/* udx check */
#define UC(F) do {                              \
        error_stack_push(__FILE__, __LINE__);   \
        F;                                      \
        if (error_get()) {                      \
            error_report();                     \
            error_abort();                      \
        }                                       \
        error_stack_pop();                      \
    } while (0)

/* [B]o[p] [c]heck */
#define BPC(ans) do {                           \
        BopStatus s = (ans);                    \
        if (!bop_success(s)) {                  \
            ERR(":%s:%d: %s\n%s\n",             \
                __FILE__, __LINE__,             \
                bop_report_error_desc(s),       \
                bop_report_error_mesg());       \
        }} while (0)

void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
/* end src/utils/error.h */
/* begin src/utils/imp.h */
// tag::mem[]
void emalloc(size_t, /**/ void**); // <1>
void efree(void*); // <2>
void *ememcpy(void *dest, const void *src, size_t n); // <3>
// end::mem[]

// tag::macros[]
#define EMALLOC(n, ppdata) UC(emalloc((n)*sizeof(**(ppdata)), (void**)(ppdata)))
#define EFREE(pdata) UC(efree(pdata))
#define EMEMCPY(n, src, dest) ememcpy((void*)(dest), (const void*)(src), (n)*sizeof(*(dest)))
// end::macros[]

// tag::stdio[]
void efopen(const char *fname, const char *mode, /**/ FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);
// end::stdio[]

// tag::tools[]
bool same_str(const char *a, const char *b); // <1>
const char* get_filename_ext(const char*);
// end::tools[]
/* end src/utils/imp.h */

/* begin src/utils/msg.h */
// tag::int[]
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
// end::int[]
/* end src/utils/msg.h */

#ifndef UDX_RBC_DEVICE_API_COMMON
#define UDX_RBC_DEVICE_API_COMMON
/* begin src/d/api.h */
#ifndef UDX_RBC_CORE_DEVICE_API_COMMON
#define UDX_RBC_CORE_DEVICE_API_COMMON
namespace d {  /* a wrapper for device API */
typedef int Stream_t; /* TODO: streams are not supported */
enum {MemcpyHostToHost, MemcpyHostToDevice,
      MemcpyDeviceToHost, MemcpyDeviceToDevice,
      MemcpyDefault};

// tag::more[]
const char *emsg();
int alloc_pinned(void **pHost, size_t size);
int is_device_pointer(const void *ptr);
// end::more[]

// tag::api[]
int GetSymbolAddress(void **devPtr, const void *symbol);
int Malloc(void **p, size_t);
int MemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset=0, int kind=MemcpyHostToDevice);
int MemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset=0, int kind=MemcpyDeviceToHost);
int HostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags);
int Memcpy (void *dst, const void *src, size_t count, int kind);
int MemsetAsync (void *devPtr, int value, size_t count, Stream_t stream=0);
int Memset (void *devPtr, int value, size_t count);
int MemcpyAsync (void * dst, const void * src, size_t count, int kind, Stream_t stream = 0);
int Free (void *devPtr);
int FreeHost (void *hstPtr);
int DeviceSynchronize (void);
int PeekAtLastError(void);
// end::api[]
}
/* end src/d/api.h */
#endif
#endif
/* begin src/utils/cc.h */
/* [c]uda [c]heck macros */

#if   defined(CC_SYNC)
/* begin src/utils/cc/sync.h */
/* Aggressive [sync]ronization */
#define CC(ans)                                             \
    do {                                                    \
        d::DeviceSynchronize();                             \
        cc::check((ans), __FILE__, __LINE__);               \
        d::DeviceSynchronize();                             \
    } while (0)
/* end src/utils/cc/sync.h */
#elif defined(CC_RELEASE)
/* begin src/utils/cc/release.h */
/* [c]cuda [c]heck */
#define CC(ans)                                             \
    do { cc::check((ans), __FILE__, __LINE__);} while (0)
/* end src/utils/cc/release.h */
#else
  #error CC_* is undefined
#endif

/* begin src/utils/cc/common.h */
namespace cc {
void check(int rc, const char *file, int line);
}


/* end src/utils/cc/common.h */

/* TODO: header includes header */
/* end src/utils/cc.h */
/* begin src/inc/dev.h */
/* ceiling `m' to `n' (returns the smallest `A' such n*A is not less than `m') */
#define ceiln(m, n) (   ((m) + (n) - 1)/(n)   )

/* a common kernel execution configuration */
#define k_cnf(n) ceiln((n), 128), 128

#define dSync() CC(d::DeviceSynchronize())

#define D2D d::MemcpyDeviceToDevice
#define D2H d::MemcpyDeviceToHost
#define H2D d::MemcpyHostToDevice
#define H2H d::MemcpyHostToHost
#define A2A d::MemcpyDefault /* "[a]ny to [a]ny" */

#define cD2D(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), D2D))
#define cH2H(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), H2H))  /* [t]to, [f]rom */
#define cA2A(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), A2A))
#define cD2H(H, D, n) CC(d::Memcpy((H), (D), (n) * sizeof((H)[0]), D2H))
#define cH2D(D, H, n) CC(d::Memcpy((D), (H), (n) * sizeof((H)[0]), H2D))

#define aD2D(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), D2D))
#define aH2H(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), H2H))  /* [t]to, [f]rom */
#define aA2A(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), A2A))
#define aD2H(H, D, n) CC(d::MemcpyAsync((H), (D), (n) * sizeof((H)[0]), D2H))
#define aH2D(D, H, n) CC(d::MemcpyAsync((D), (H), (n) * sizeof((H)[0]), H2D))

/* device allocation */
#define Dfree(D)      CC(d::Free(D))
#define Dalloc(D, n)  CC(d::Malloc((void**)(D), (n) * sizeof(**(D))))

/* [d]evice set */
#define Dset(P, v, n) CC(d::Memset(P, v, (n)*sizeof(*(P))))
#define Dzero(P, n)   Dset(P, 0, n)

/* [d]evice [a]synchronous set */
#define DsetA(P, v, n) CC(d::MemsetAsync(P, v, (n)*sizeof(*(P))))
#define DzeroA(P, n)   DsetA(P, 0, n)
/* end src/inc/dev.h */

/* begin src/rbc/adj/imp.h */
struct Adj;
struct Adj_v;
struct AdjMap;
struct int4;

// tag::interface[]
void adj_ini(int md, int nt, int nv, const int4 *tt, /**/ Adj**);
void adj_fin(Adj*);
int  adj_get_map(int i, const Adj*, /**/ AdjMap *m);
int  adj_get_max(const Adj*);
void adj_get_anti(const Adj*, /**/ int *anti);

void adj_view_ini(const Adj*, Adj_v**);
void adj_view_fin(Adj_v*);
// end::interface[]
/* end src/rbc/adj/imp.h */

/* begin src/algo/edg/imp.h */
// tag::interface[]
void edg_ini(int md, int nv,                     int *hx);
void edg_set(int md, int i, int j, int val, /**/ int *hx,       int *hy); // <1>
int  edg_get(int md, int i, int j,         const int *hx, const int *hy); // <2>
int  edg_valid(int md, int i, int j,         const int *hx); // <3>
// end::interface[]
/* end src/algo/edg/imp.h */

/* begin src/rbc/adj/type/common.h */
#ifndef UDX_RBC_ADJ_MAP_TYPE
#define UDX_RBC_ADJ_MAP_TYPE
// tag::interface[]
struct AdjMap {
    int i0, i1, i2, i3, i4;
    int rbc;
};
// end::interface[]
/* end src/rbc/adj/type/common.h */
#endif
/* begin src/rbc/adj/type/dev.h */
struct Adj_v {
    int *adj0;
    int *adj1;
    int nv, md;
};

/* end src/rbc/adj/type/dev.h */

/* begin src/rbc/adj/imp/type.h */
struct Adj { /* adjacency lists */
    int *adj0;
    int *adj1;
    int nv, md;
};
/* end src/rbc/adj/imp/type.h */
/* begin src/rbc/adj/imp/ini.h */
static void gen_a12(int md, int i0, int *hx, int *hy, /**/ int *a1, int *a2) {
    int lo = i0*md, hi = lo + md, mi = hx[lo];
    int i;
    for (i = lo + 1; (i < hi) && (hx[i] != -1); i++)
        if (hx[i] < mi) mi = hx[i]; /* minimum */

    int c = mi, c0;
    i = lo;
    do {
        c     = edg_get(md, i0, c0 = c, hx, hy);
        a1[i] = c0;
        a2[i] = edg_get(md, c, c0, hx, hy);
        i++;
    }  while (c != mi);
}

static void ini0(int md, int nt, int nv, const int4 *faces, /**/ int *a1, int *a2) {
    int hx[nv*md], hy[nv*md];
    int i;
    for (i = 0; i < nv*md; i++) a1[i] = a2[i] = -1;
    edg_ini(nv, md, hx);

    int4 t;
    for (int ifa = 0; ifa < nt; ifa++) {
        t = faces[ifa];
        int f0 = t.x, f1 = t.y, f2 = t.z;
        edg_set(md, f0, f1, f2,   hx, hy); /* register an edge */
        edg_set(md, f1, f2, f0,   hx, hy);
        edg_set(md, f2, f0, f1,   hx, hy);
    }
    for (i = 0; i < nv; i++) gen_a12(md, i, hx, hy, /**/ a1, a2);
}

void adj_ini(int md, int nt, int nv, const int4 *faces, /**/ Adj **pq) {
    int *a1, *a2;
    Adj *q;
    EMALLOC(1    , &q);
    EMALLOC(md*nv, &q->adj0);
    EMALLOC(md*nv, &q->adj1);
    a1 = q->adj0; /* sic */
    a2 = q->adj1;
    q->md = md; q->nv = nv;
    ini0(md, nt, nv, faces, /**/ a1, a2);
    *pq = q;
}

void adj_view_ini(const Adj *hst, /**/ Adj_v **pq) {
    int nv, md;
    Adj_v *q;
    if (hst == NULL) ERR("hst == NULL");
    nv = hst->nv; md = hst->md;
    EMALLOC(1, &q);
    Dalloc(&q->adj0, md*nv);
    Dalloc(&q->adj1, md*nv);

    q->nv = nv; q->md = md;
    cH2D(q->adj0, hst->adj0, md*nv);
    cH2D(q->adj1, hst->adj1, md*nv);

    *pq = q;
}
/* end src/rbc/adj/imp/ini.h */
/* begin src/rbc/adj/imp/fin.h */
void adj_fin(Adj *q) {
    EFREE(q->adj0);
    EFREE(q->adj1);
    EFREE(q);
}

void adj_view_fin(Adj_v *q) {
    Dfree(q->adj0);
    Dfree(q->adj1);
    EFREE(q);
}
/* end src/rbc/adj/imp/fin.h */
/* begin src/rbc/adj/imp/map.h */
static int fetch(const int *adj, int i) { return adj[i]; }
static int hst0(int md, int nv, int i, const int *adj0, const int *adj1, /**/ AdjMap *m) {
    int i0, i1, i2, i3, i4;
    int j, k;
    if (i >= md * nv)
        ERR("(i = %d) >= (md = %d) * (nv = %d)", i, md, nv);

    i0 = i / md;
    j  = i % md;

    k   = i0 * md;
    i1 = fetch(adj0, k + j);
    if (i1 == -1) return 0; /* invalid */

    i2 = fetch(adj0, k + ((j + 1) % md));
    if (i2 == -1) {
        i2 = fetch(adj0, k    );
        i3 = fetch(adj0, k + 1);
    } else {
        i3 = fetch(adj0, k + ((j + 2) % md));
        if (i3 == -1) i3 = fetch(adj0, k);
    }
    i4 = fetch(adj1, k + j);

    m->i0 = i0; m->i1 = i1; m->i2 = i2; m->i3 = i3; m->i4 = i4;
    return 1;
}

int adj_get_map(int i, const Adj *q, /**/ AdjMap *m) {
    int md, nv;
    const int *adj0, *adj1;
    md = q->md;
    nv = q->nv;
    adj0 = q->adj0;
    adj1 = q->adj1;
    return hst0(md, nv, i, adj0, adj1, m);
}

int adj_get_max(const Adj *q) { return (q->md) * (q->nv); }
/* end src/rbc/adj/imp/map.h */
/* begin src/rbc/adj/imp/anti.h */
static void set(int md, int nv, const Adj *adj, /**/ int *hx, int *hy) {
    AdjMap m;
    int valid, i, i0, i1;
    for (i = 0; i < md*nv; i++) {
        valid = adj_get_map(i, adj, /**/ &m);
        if (!valid) continue;
        i0 = m.i0; i1 = m.i1;
        edg_set(md, i0, i1, i,  hx, hy);
    }
}

static void get(int md, int nv, const Adj *adj, const int *hx, const int *hy, /**/ int *anti) {
    AdjMap m;
    int valid, i, j, i0, i1;
    for (i = 0; i < md*nv; i++) {
        valid = adj_get_map(i, adj, /**/ &m);
        if (!valid) continue;
        i0 = m.i0; i1 = m.i1;
        /* invert i1 and i0 */
        j = edg_get(md, i1, i0, hx, hy);
        anti[i] = j;
    }
}

static void ini0(int md, int nv, const Adj *adj, /**/ int *anti, /*w*/ int *hx, int *hy) {
    edg_ini(md, nv, /**/ hx);
    set(md, nv, adj, /**/ hx, hy);
    get(md, nv, adj, hx, hy, /**/ anti);
}

void adj_get_anti(const Adj *adj, /**/ int *anti) {
    int md, nv, n;
    int *hx, *hy;
    md = adj->md;
    nv = adj->nv;
    n = md*nv;
    EMALLOC(n, &hx);
    EMALLOC(n, &hy);
    ini0(md, nv, adj, /**/ anti, /*w*/ hx, hy);
    EFREE(hx); EFREE(hy);
}
/* end src/rbc/adj/imp/anti.h */
/* end src/rbc/adj/imp.cpp */

#line 1 "src/rbc/com/imp.cu"
