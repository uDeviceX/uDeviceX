// generated from udx.flat.cu
#include "udx.config.cuh"
#include "udx.mesh-force-kantor-common.cuh"

/* begin src/mesh/force/kantor0/imp.cpp */
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <assert.h>
#include <vector_types.h>

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
/* begin src/mesh/force/kantor0/imp.h */
struct float3;
struct double3;
namespace force_kantor0_hst {
// tag::interface[]
double3 dih_a(double phi, double kb, double3 a, double3 b, double3 c, double3 d);
double3 dih_b(double phi, double kb, double3 a, double3 b, double3 c, double3 d);
// end::interface[]
}
/* end src/mesh/force/kantor0/imp.h */

#define FORCE_KANTOR0_HOST
/* begin src/mesh/force/kantor0/dev.h */
#ifdef FORCE_KANTOR0_HOST
  #define _I_
  #define _S_ static
  #define BEGIN namespace force_kantor0_hst {
  #define END }
#else
  #define _I_ static __device__
  #define _S_ static __device__
  #define BEGIN namespace force_kantor0_dev {
  #define END }
#endif

BEGIN

#ifdef FORCE_KANTOR0_HOST
_S_ double rsqrt0(double x) { return pow(x, -0.5); }
#define PRINT(fmt, ...) msg_print((fmt), ##__VA_ARGS__)
#define EXIT() ERR("assert")
#else
_S_ double rsqrt0(double x) { return rsqrt(x); }
#define PRINT(fmt, ...) printf((fmt), ##__VA_ARGS__)
#define EXIT() assert(0)
#endif

_S_ double max(double a, double b) { return a > b ? a : b; }

enum {ANGLE_OK, ANGLE_BAD_A, ANGLE_BAD_B};
_S_ int angle(double3 a, double3 b, /**/ double *pcos, double *pover_sin,
              double *pover_a, double *pover_b) {
    double  cost, over_sin, over_a, over_b;
    double aa, bb, ab;
    *pcos = *pover_sin = *pover_a = *pover_b = 0.0;

    aa = dot<double>(&a, &a);
    bb = dot<double>(&b, &b);
    ab = dot<double>(&a, &b);

    if (aa == 0) return ANGLE_BAD_A;
    if (bb == 0) return ANGLE_BAD_B;

    over_a = rsqrt0(aa);
    over_b = rsqrt0(bb);
    cost = ab * over_a * over_b;
    over_sin = rsqrt0(max(1.0 - cost*cost, 1e-6));

    *pcos = cost; *pover_sin = over_sin; *pover_a = over_a; *pover_b = over_b;
    return ANGLE_OK;
}

_S_ void dih_write(double3 a, double3 b, double3 c, double3 d) {
    PRINT("%g %g %g\n", a.x, a.y, a.z);
    PRINT("%g %g %g\n", b.x, b.y, b.z);
    PRINT("%g %g %g\n", c.x, c.y, c.z);
    PRINT("%g %g %g\n", d.x, d.y, d.z);
}

/* forces from one dihedral */
template <int update>
_S_ double3 dih0(double phi, double kb,
                 double3 a, double3 b, double3 c, double3 d) {
    int status;
    double over_k, over_n, cosTheta, over_sin,
        beta, b11, b12, sint0kb, cost0kb;
    double3 ab, ac, cd, bc, bd, da, k, n, k_n;
    diff(&a, &b, /**/ &ab);
    diff(&a, &c, /**/ &ac);
    diff(&c, &d, /**/ &cd);
    diff(&b, &d, /**/ &bd);
    diff(&b, &c, /**/ &bc);
    diff(&d, &a, /**/ &da);

    cross(&ab, &ac, /**/ &k);
    cross(&bd, &bc, /**/ &n);

    status = angle(k, n, /**/
                   &cosTheta, &over_sin,
                   &over_k, &over_n);
    if (status != ANGLE_OK) {
        PRINT("bad dihedral\n");
        dih_write(a, b, c, d);
        EXIT();
    }
    diff(&k, &n, /**/ &k_n);
    over_sin = copysign(over_sin, dot<double>(&k_n, &da));

    sint0kb = sin(phi) * kb;
    cost0kb = cos(phi) * kb;
    beta = cost0kb - cosTheta * sint0kb * over_sin;

    b11 = -beta *  cosTheta * over_k * over_k;
    b12 =  beta *             over_k * over_n;

    if (update == 1) {
        double3 r32, f1, f;
        diff(&c, &b, /**/ &r32);
        cross(&k, &r32, /**/ &f);
        cross(&n, &r32, /**/ &f1);
        scal(b11, /**/ &f);
        axpy(b12, &f1, /**/ &f);
        return f;
    }
    else if (update == 2) {
        double3 f, f1, f2, f3;
        double b22 = -beta * cosTheta * over_n * over_n;

        cross(&k, &ac, /**/ &f);
        cross(&k, &cd, /**/ &f1);
        cross(&n, &ac, /**/ &f2);
        cross(&n, &cd, /**/ &f3);

        scal(b11, /**/ &f);
        add(&f2, /**/ &f1);
        axpy(b12, &f1, /**/ &f);
        axpy(b22, &f3, /**/ &f);
        return f;
    }
    else {
        double3 f;
        f.x = f.y = f.z = 0;
        assert(0);
        return f;
    }
}

_I_ double3 dih_a(double phi, double kb, double3 a, double3 b, double3 c, double3 d) {
    return dih0<1>(phi, kb, a, b, c, d);
}

_I_ double3 dih_b(double phi, double kb, double3 a, double3 b, double3 c, double3 d) {
    return dih0<2>(phi, kb, a, b, c, d);
}

END

#undef _I_
#undef _S_
#undef BEGIN
#undef END
/* end src/mesh/force/kantor0/dev.h */
#undef FORCE_KANTOR0_HOST
/* end src/mesh/force/kantor0/imp.cpp */

#line 1 "src/mesh/force/kantor1/imp.cpp"
