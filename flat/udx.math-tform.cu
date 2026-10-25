// generated from udx.flat.cu
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include "udx.api.cuh"

/* begin src/math/tform/imp.cpp */
#include <stdio.h>

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

/* begin src/math/tform/type.h */
struct Tform_v { float o[3], s[3]; };
/* end src/math/tform/type.h */
/* begin src/math/tform/imp.h */
struct Tform;
struct Tform_v;

// tag::ini[]
void tform_ini(Tform**);
void tform_fin(Tform*);
// end::ini[]

// tag::def[]
void tform_vector(const float a0[3], const float a1[3],   const float b0[3], const float b1[3], /**/ Tform*); // <1>
void tform_chain(const Tform *A, const Tform *B, /**/ Tform *C); // <2>
void tform_to_grid(const float lo[3], const float hi[3], const int n[3], /**/ Tform*); // <3>
void tform_from_grid(const float a0[3], const float b0[3], const int n[3], /**/ Tform*); // <4>
void tform_grid2grid(const float f_lo[3], const float f_hi[3], const int f_n[3],
                     const float t_lo[3], const float t_hi[3], const int t_n[3],
                     Tform*); // <5>
// end::def[]


// tag::convert[]
void tform_convert(const Tform*, const float a0[3], /**/ float a1[3]);
// end::convert[]

// tag::view[]
void tform_to_view(const Tform*, /**/ Tform_v*);
// end::view[]

// tag::log[]
void tform_log(Tform*);
void tform_dump(Tform*, FILE*);
// end::log[]
/* end src/math/tform/imp.h */

/* begin src/math/tform/imp/type.h */
struct Tform   { float o[3], s[3]; };
/* end src/math/tform/imp/type.h */
/* begin src/math/tform/imp/main.h */
void tform_ini(Tform **pq) { EMALLOC(1, pq); }
void tform_fin(Tform *q)   { EFREE(q);       }

static void report(const float a0[3], const float a1[3],   const float b0[3], const float b1[3]) {
    enum {X, Y, Z};
    msg_print("a0: %g %g %g", a0[X], a0[Y], a0[Z]);
    msg_print("a1: %g %g %g", a1[X], a1[Y], a1[Z]);
    msg_print("b0: %g %g %g", b0[X], b0[Y], b0[Z]);
    msg_print("b1: %g %g %g", b1[X], b1[Y], b1[Z]);
}
enum {OK, ZERO};
static int os(float a0, float a1,   float b0, float b1, /**/ float *o, float *s) {
    /* origin, space */
    const float eps = 1e-8;
    float d;
    d = b0 - a0;
    if (-eps < d && d < eps) return ZERO;
    *o = (a1*b0-a0*b1)/(b0-a0);
    *s = (b1-a1)      /(b0-a0);
    return OK;
}
void tform_vector(const float a0[3], const float a1[3],   const float b0[3], const float b1[3], /**/ Tform* t) {
    int r;
    enum {X, Y, Z};
    r = os(a0[X], a1[X], b0[X], b1[X], /**/ &t->o[X], &t->s[X]);
    if (r != OK) { report(a0, a1,   b0, b1); ERR("tform_vector failed"); }

    r = os(a0[Y], a1[Y], b0[Y], b1[Y], /**/ &t->o[Y], &t->s[Y]);
    if (r != OK) { report(a0, a1,   b0, b1); ERR("tform_vector failed"); }

    r = os(a0[Z], a1[Z], b0[Z], b1[Z], /**/ &t->o[Z], &t->s[Z]);
    if (r != OK) { report(a0, a1,   b0, b1); ERR("tform_vector failed"); }
}

static int smallp(const float s[3]) {
    enum {X, Y, Z};
    const float eps = 1e-12;
    int cx, cy, cz;
    cx = -eps < s[X] && s[X] < eps;
    cy = -eps < s[Y] && s[Y] < eps;
    cz = -eps < s[Z] && s[Z] < eps;
    return cx && cy && cz;
}
void tform_convert(const Tform *t, const float a0[3], /**/ float a1[3]) {
    enum {X, Y, Z};
    const float *o, *s;
    o = t->o; s = t->s;
    if (smallp(s))
        ERR("tform_convert failed: s = [%g %g %g]\n", s[X], s[Y], s[Z]);
    a1[X] = s[X]*a0[X] + o[X];
    a1[Y] = s[Y]*a0[Y] + o[Y];
    a1[Z] = s[Z]*a0[Z] + o[Z];
}

void tform_chain(const Tform *t1, const Tform *t2, /**/ Tform *t) {
    float a0[3] = {0, 0, 0};
    float b0[3] = {1, 1, 1};
    float a1[3], b1[3], a2[3], b2[3];

    tform_convert(t1, a0, /**/ a1);
    tform_convert(t1, b0, /**/ b1);

    tform_convert(t2, a1, /**/ a2);
    tform_convert(t2, b1, /**/ b2);

    UC(tform_vector(a0, a2, b0, b2, t));
}

void tform_log(Tform *t) {
    enum {X, Y, Z};
    float *o, *s;
    o = t->o; s = t->s;
    msg_print("tform: o = [%g %g %g]", o[X], o[Y], o[Z]);
    msg_print("tform: s = [%g %g %g]", s[X], s[Y], s[Z]);
}

void tform_dump(Tform *t, FILE *f) {
    enum {X, Y, Z};
    float *o, *s;
    o = t->o; s = t->s;
    fprintf(f, "%16.10e %16.10e %16.10e\n", o[X], o[Y], o[Z]);
    fprintf(f, "%16.10e %16.10e %16.10e\n", s[X], s[Y], s[Z]);
}

void tform_to_grid(const float a0[3], const float b0[3], const int n[3], /**/ Tform *t) {
    enum {X, Y, Z};
    float a1[3], b1[3];
    a1[X] = a1[Y] = a1[Z] = -0.5;
    b1[X] = n[X] - 0.5; b1[Y] = n[Y] - 0.5; b1[Z] = n[Z] - 0.5;
    UC(tform_vector(a0, a1,   b0, b1, /**/ t));
}
void tform_from_grid(const float a0[3], const float b0[3], const int n[3], /**/ Tform *t) {
    enum {X, Y, Z};
    float a1[3], b1[3];
    a1[X] = a1[Y] = a1[Z] = -0.5;
    b1[X] = n[X] - 0.5; b1[Y] = n[Y] - 0.5; b1[Z] = n[Z] - 0.5;
    UC(tform_vector(a1, a0,   b1, b0, /**/ t)); /* inverted */
}
enum {LH_OK, LH_BAD};
static int assert_lh(const float lo[3], const float hi[3]) {
    enum {X, Y, Z};
    int c;
    c = lo[X] < hi[X] && lo[X] < hi[X] && lo[X] < hi[X];
    return c ? LH_OK : LH_BAD;
}
void tform_grid2grid(const float f_lo[3], const float f_hi[3], const int f_n[3],
                     const float t_lo[3], const float t_hi[3], const int t_n[3], /**/
                     Tform *t) {
    enum {X, Y, Z};
    /* f: from, t: to, g: global */
    Tform *f2g, *g2t;
    if (assert_lh(f_lo, f_hi) == LH_BAD)
        ERR("wrong f_[lo|hi]: [%g %g %g] [%g %g %g]",
            f_lo[X], f_lo[Y], f_lo[X], f_hi[X], f_hi[Y], f_hi[Z]);

    if (assert_lh(t_lo, t_hi) == LH_BAD)
        ERR("wrong f_[lo|hi]: [%g %g %g] [%g %g %g]",
            t_lo[X], t_lo[Y], t_lo[X], t_hi[X], t_hi[Y], t_hi[Z]);

    UC(tform_ini(&f2g));
    UC(tform_ini(&g2t));

    UC(tform_from_grid(f_lo, f_hi, f_n, /**/ f2g));
    UC(tform_to_grid  (t_lo, t_hi, t_n, /**/ g2t));
    UC(tform_chain(f2g, g2t, /**/ t));

    tform_fin(g2t);
    tform_fin(f2g);
}

void tform_to_view(const Tform *t, Tform_v *v) {
    enum {X, Y, Z};
    if (smallp(t->s))
        ERR("tform_to_view failed: t->s = [%g %g %g]\n", t->s[X], t->s[Y], t->s[Z]);

    v->o[X] = t->o[X];
    v->o[Y] = t->o[Y];
    v->o[Z] = t->o[Z];

    v->s[X] = t->s[X];
    v->s[Y] = t->s[Y];
    v->s[Z] = t->s[Z];
}
/* end src/math/tform/imp/main.h */
/* end src/math/tform/imp.cpp */

#line 1 "src/math/tri/imp.cpp"
