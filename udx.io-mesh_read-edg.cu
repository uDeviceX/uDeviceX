// generated from udx.flat.cu
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include "udx.api.cuh"

/* begin src/io/mesh_read/edg/imp.cpp */
#include <stdio.h>

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

/* begin src/algo/edg/imp.h */
// tag::interface[]
void edg_ini(int md, int nv,                     int *hx);
void edg_set(int md, int i, int j, int val, /**/ int *hx,       int *hy); // <1>
int  edg_get(int md, int i, int j,         const int *hx, const int *hy); // <2>
int  edg_valid(int md, int i, int j,         const int *hx); // <3>
// end::interface[]
/* end src/algo/edg/imp.h */

/* begin src/io/mesh_read/edg/imp/type.h */
struct Edg {
    int md, nv;
    int *hx, *hy;
};
/* end src/io/mesh_read/edg/imp/type.h */
/* begin src/io/mesh_read/edg/imp/main.h */
void e_ini(int md, int nv, /**/ Edg **pq) {
    Edg *q;
    EMALLOC(1, &q);
    EMALLOC(md*nv, &q->hx);
    EMALLOC(md*nv, &q->hy);
    edg_ini(md, nv, /**/ q->hx);
    q->md = md; q->nv = nv;
    *pq = q;
}

void e_fin(Edg *q) {
    EFREE(q->hx);
    EFREE(q->hy);
    EFREE(q);
}

void e_set(Edg *q, int i, int j, int val) {
    int md, *hx, *hy;
    md = q->md; hx = q->hx; hy = q->hy;
    UC(edg_set(md, i, j, val, hx, hy));
}

int e_get(Edg *q, int i, int j) {
    int md, ans;
    const int *hx, *hy;
    md = q->md; hx = q->hx; hy = q->hy;
    UC(ans = edg_get(md, i, j, hx, hy));
    return ans;
}

int e_valid(Edg *q, int i, int j) {
    int md;
    const int *hx;
    md = q->md; hx = q->hx;
    return edg_valid(md, i, j, hx);
}
/* end src/io/mesh_read/edg/imp/main.h */
/* end src/io/mesh_read/edg/imp.cpp */

#line 1 "src/io/mesh_read/imp.cpp"
