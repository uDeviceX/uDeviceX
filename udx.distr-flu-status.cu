// generated from udx.flat.cu
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include "udx.api.cuh"

/* begin src/distr/flu/status/imp.cpp */
#include <stdio.h>
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
/* begin src/utils/msg.h */
// tag::int[]
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
// end::int[]
/* end src/utils/msg.h */
/* begin src/frag/imp.h */
#ifndef UDX_DISTR_EXCH_FRAG_IMP_COMMON
#define UDX_DISTR_EXCH_FRAG_IMP_COMMON
enum {frag_bulk = 26};
#endif

struct int3;

namespace frag_hst {

// tag::i2d[]
int i2dx(int i);                 // <1>
int i2dy(int i);                 // <2>
int i2dz(int i);                 // <3>
void i2d3(int i, /**/ int d[3]); // <4>
// end::i2d[]

// tag::d2i[]
int d2i(int x, int y, int z); // <1>
int d32i(const int d[3]);     // <2>
// end::d2i[]

// tag::ncell[]
int ncell(int3 L, int i); // <1>
// end::ncell[]

// tag::anti[]
int antid2i(int x, int y, int z); // <1>
int anti(int i);                  // <2>
// end::anti[]

// tag::hst[]
void estimates(int3 L, int nfrags, float maxd, /**/ int *cap); // <1>
// end::hst[]

} // frag_hst
/* end src/frag/imp.h */

/* begin src/distr/flu/status/imp.h */
struct DFluStatus;
struct Config;

/* ini */
void dflu_status_ini(DFluStatus **s);
void dflu_status_fin(DFluStatus  *s);

/* get */
int  dflu_status_nullp(DFluStatus *s);
int  dflu_status_success(DFluStatus *s);
void dflu_status_log(DFluStatus *s);

/* set */
void dflu_status_exceed(int frag, int counter, int capacity, /**/ DFluStatus *s);

/* end src/distr/flu/status/imp.h */

enum {SUCCESS, PACK_FAILURE};
struct DFluStatus {
    int errorcode;
    int cap, cnt, fid; /* capacity, count, fragment */
};

void dflu_status_ini(DFluStatus **ps) {
    DFluStatus *s;
    EMALLOC(1, &s);
    s->errorcode = SUCCESS;
    *ps = s;
}

void dflu_status_fin(DFluStatus *s) {
    EFREE(s);
}

int  dflu_status_success(DFluStatus *s) {
    return s->errorcode == SUCCESS;
}

static void pack_failure(DFluStatus *s) {
    enum {X, Y, Z};
    int cap, cnt, fid, d[3];
    cap = s->cap; cnt = s->cnt; fid = s->fid;
    d[X] = frag_hst::i2dx(fid);
    d[Y] = frag_hst::i2dy(fid);
    d[Z] = frag_hst::i2dz(fid);
    msg_print("exceed capacity, fragment %d = [%d %d %d]: %d/%d",
              fid, d[X], d[Y], d[Z], cnt, cap);
}
void dflu_status_log(DFluStatus *s) {
    int code;
    if (s == NULL) msg_print("DFluStatus: s == NULL");
    else {
        code = s->errorcode;
        if      (code == SUCCESS)      msg_print("DFluStatus: SUCCESS");
        else if (code == PACK_FAILURE) pack_failure(s);
        else ERR("unknown errorcode = %d\n", code);
    }
}

void dflu_status_exceed(int fid, int cnt, int cap, /**/ DFluStatus *s) {
    if (s == NULL) ERR("status == NULL");
    s->fid = fid; s->cnt = cnt; s->cap = cap;
    s->errorcode = PACK_FAILURE;
}

int dflu_status_nullp(DFluStatus *s) { return s == NULL; }
/* end src/distr/flu/status/imp.cpp */

#line 1 "src/distr/map/imp.cu"
