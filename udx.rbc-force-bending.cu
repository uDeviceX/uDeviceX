// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/rbc/force/bending/imp.cpp */
#include <stdio.h>
#include <assert.h>
#include <stddef.h>

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
/* begin src/inc/macro.h */
#define CONTAINER_OF(ptr, type, member) \
    ((type*)((char*)(ptr) - offsetof(type, member)))
/* end src/inc/macro.h */

/* begin src/inc/def.h */
/* solvent colors */
#define XMACRO_COLOR(_)                         \
    _(BLUE)                                     \
    _(RED)

#define ENUM_COLOR(a) a##_COLOR
#define make_enum_color(a) ENUM_COLOR(a) ,
enum {
    XMACRO_COLOR(make_enum_color)
    N_COLOR
};
#undef make_enum_color

enum {
    /* wall margins */
    XWM = 6,
    YWM = 6,
    ZWM = 6,

    SAFETY_FACTOR_MAXP = 3,

    /* maximum number of particles per solid */
    MAX_PSOLID_NUM = 12000,

    /* maximum number of solids per node */
    MAX_SOLIDS = 100,

    /* maximum number of object types (solid, rbc, ...) */
    MAX_MBR_TYPES = 2,
    MAX_RIG_TYPES = 2,
    MAX_OBJ_TYPES = MAX_MBR_TYPES + MAX_RIG_TYPES,

    /* maximum number density of particles of the objects */
    MAX_OBJ_DENSITY = 30,

    /* maximum number of red blood cells per node */
    MAX_CELL_NUM = 1500,

    /* maximum texture size in bytes */
    MAX_TEXO_SIZE = 100000000,

    /* safety factor for dpd halo interactions */
    HSAFETY_FACTOR = 10,

    /* safety factor for odist fragments */
    ODSTR_FACTOR = 3,
};

/* for the spring forces r = min(lmax*RBC_SPRING_CAP, r) */
#define RBC_SPRING_CAP (0.99)
/* end src/inc/def.h */

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

/* begin src/rbc/force/bending/kantor/imp.h */
struct RbcParams;
struct Kantor;
struct Force;
struct MeshRead;
struct RbcQuants;

// tag::mem[]
void kantor_ini(const MeshRead*, Kantor**);
void kantor_fin(Kantor*);
// end::mem[]

// tag::apply[]
void kantor_apply(Kantor*, const RbcParams*, const RbcQuants*, /**/ Force*); // <1>
// end::apply[]
/* end src/rbc/force/bending/kantor/imp.h */
/* begin src/rbc/force/bending/juelicher/imp.h */
struct RbcParams;
struct Juelicher;
struct Force;
struct MeshRead;
struct RbcQuants;

// tag::mem[]
void juelicher_ini(const MeshRead*, Juelicher**);
void juelicher_fin(Juelicher*);
// end::mem[]

// tag::apply[]
void juelicher_apply(Juelicher*, const RbcParams*, const RbcQuants*, /**/ Force*); // <1>
// end::apply[]
/* end src/rbc/force/bending/juelicher/imp.h */

/* begin src/rbc/force/bending/imp.h */
struct RbcParams;
struct Bending;
struct Force;
struct MeshRead;
struct RbcQuants;

// tag::mem[]
void bending_kantor_ini(const MeshRead*, Bending**);
void bending_juelicher_ini(const MeshRead*, Bending**);
void bending_none_ini(const MeshRead*, Bending**);

void bending_fin(Bending*);
// end::mem[]

// tag::apply[]
void bending_apply(Bending*, const RbcParams*, const RbcQuants*, /**/ Force*); // <1>
// end::apply[]
/* end src/rbc/force/bending/imp.h */
/* begin src/rbc/force/bending/imp/main.h */
struct Bending {
    struct Bending_vtable *vtable;
};

struct Bending_vtable {
    void (*apply)(Bending*, const RbcParams*, const RbcQuants*, /**/ Force*);
    void (*fin)(Bending*);
};

void bending_apply(Bending *q, const RbcParams *params, const RbcQuants *quants, /**/ Force *ff) {
    q->vtable->apply(q, params, quants, /**/ ff);
}

void bending_fin(Bending *q) {
    q->vtable->fin(q);
}

struct BendingKantor {
    Bending bending;
    Kantor *kantor;
};
static void method_kantor_apply(Bending *q, const RbcParams *params, const RbcQuants *quants, /**/ Force *ff) {
    BendingKantor *b = CONTAINER_OF(q, BendingKantor, bending);
    UC(kantor_apply(b->kantor, params, quants, ff));
}
static void method_kantor_fin(Bending *q) {
    BendingKantor *b = CONTAINER_OF(q, BendingKantor, bending);
    UC(kantor_fin(b->kantor));
    EFREE(q);
}
static Bending_vtable BendingKantor_vtable = { method_kantor_apply, method_kantor_fin};
void bending_kantor_ini(const MeshRead *cell, /**/ Bending **pq) {
    BendingKantor *q;
    EMALLOC(1, &q);
    UC(kantor_ini(cell, &q->kantor));
    q->bending.vtable = &BendingKantor_vtable;
    *pq = &q->bending;
}


struct BendingJuelicher {
    Bending bending;
    Juelicher *juelicher;
};
static void method_juelicher_apply(Bending *q, const RbcParams *params, const RbcQuants *quants, /**/ Force *ff) {
    BendingJuelicher *b = CONTAINER_OF(q, BendingJuelicher, bending);
    UC(juelicher_apply(b->juelicher, params, quants, ff));
}
static void method_juelicher_fin(Bending *q) {
    BendingJuelicher *b = CONTAINER_OF(q, BendingJuelicher, bending);
    UC(juelicher_fin(b->juelicher));
    EFREE(q);
}
static Bending_vtable BendingJuelicher_vtable = { method_juelicher_apply, method_juelicher_fin};
void bending_juelicher_ini(const MeshRead *cell, /**/ Bending **pq) {
    BendingJuelicher *q;
    EMALLOC(1, &q);
    UC(juelicher_ini(cell, &q->juelicher));
    q->bending.vtable = &BendingJuelicher_vtable;
    *pq = &q->bending;
}

struct BendingNone { Bending bending; };
static void method_none_apply(Bending*, const RbcParams*, const RbcQuants*, /**/ Force*) { }
static void method_none_fin(Bending*) { }
static Bending_vtable BendingNone_vtable = { method_none_apply, method_none_fin};
void bending_none_ini(const MeshRead*, /**/ Bending **pq) {
    BendingNone *q;
    EMALLOC(1, &q);
    q->bending.vtable = &BendingNone_vtable;
    *pq = &q->bending;
}
/* end src/rbc/force/bending/imp/main.h */
/* end src/rbc/force/bending/imp.cpp */

#line 1 "src/rbc/force/bending/juelicher/imp.cu"
