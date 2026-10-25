// generated from udx.flat.cu
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include "udx.api.cuh"

/* begin src/algo/edg/imp.cpp */
#include <stdlib.h>
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
/* begin src/algo/edg/imp.h */
// tag::interface[]
void edg_ini(int md, int nv,                     int *hx);
void edg_set(int md, int i, int j, int val, /**/ int *hx,       int *hy); // <1>
int  edg_get(int md, int i, int j,         const int *hx, const int *hy); // <2>
int  edg_valid(int md, int i, int j,         const int *hx); // <3>
// end::interface[]
/* end src/algo/edg/imp.h */

/* begin src/algo/edg/imp/main.h */
#include <stdlib.h>
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
/* begin src/fsi/imp.h */
struct RNDunif;
struct Fsi;
struct int3;

// tag::mem[]
void fsi_ini(int rank, int3 L, /**/ Fsi **fsi);
void fsi_fin(Fsi *fsi);
// end::mem[]

// tag::int[]
void fsi_bind_solvent(PaArray pa, Force *ff, int n, const int *starts, /**/ Fsi *fsi);  // <1>
void fsi_bulk(Fsi *fsi, int nw, const PairParams **prms, PaWrap *pw, FoWrap *fw); // <2>
void fsi_halo(Fsi *fsi, int nw, const PairParams **prms, Pap26 all_pp, Fop26 all_ff, const int *all_counts); // <3>
// end::int[]
/* end src/fsi/imp.h */

void edg_ini(int md, int nv, /**/ int *hx) {
    int n, i;
    n = md * nv;
    for (i = 0; i < n; i ++) hx[i] = -1;
}

void edg_set(int md, int f, int x, int y,  /**/ int *hx, int *hy) {
    int d, j;
    j = f*md; d = 0;
    while (hx[j] != -1) {
        j++; d++;
        if (d >= md) ERR("invalid edg_set call: %d/%d/%d/%d/%d", md, f, x, y, j);
    }
    hx[j] = x; hy[j] = y;
}

int edg_get(int md, int i, int x, const int *hx, const int *hy) { /* next */
    int d;
    i *= md; d = 0;
    while (hx[i] != x) {
        i++; d++;
        if (d >= md) ERR("invalid edg_get call: %d/%d/%d", md, i, x);
    }
    return hy[i];
}

int edg_valid(int md, int i, int x, const int *hx) { /* valid? */
    int d;
    i *= md; d = 0;
    while (hx[i] != x) {
        i++; d++;
        if (d >= md) return 0;
    }
    return 1;
}
/* end src/algo/edg/imp/main.h */
/* end src/algo/edg/imp.cpp */

#line 1 "src/algo/force_stat/imp.cu"
