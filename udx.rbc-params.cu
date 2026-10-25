// generated from udx.flat.cu
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include "udx.api.cuh"

/* begin src/rbc/params/conf.cpp */
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
/* begin src/conf/imp.h */
struct Config;
struct float3;
struct int3;

// tag::mem[]
void conf_ini(/**/ Config **c);
void conf_fin(/**/ Config *c);
// end::mem[]

// tag::ini[]
void conf_read(int argc, char **argv, /**/ Config *cfg);
// end::ini[]

// tag::lookup[]
void conf_lookup_int(const Config *c, const char *desc, int *a);
void conf_lookup_float(const Config *c, const char *desc, float *a);
void conf_lookup_bool(const Config *c, const char *desc, int *a);
void conf_lookup_string(const Config *c, const char *desc, const char **a);
void conf_lookup_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3(const Config *c, const char *desc, int3 *a);
void conf_lookup_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3(const Config *c, const char *desc, float3 *a);
void conf_lookup_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);
// end::lookup[]

// tag::ns[]
void conf_lookup_int_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_float_ns(const Config *c, const char *ns, const char *desc, float *a);
void conf_lookup_bool_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_string_ns(const Config *c, const char *ns, const char *desc, const char **a);
void conf_lookup_vint_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3_ns(const Config *c, const char *ns, const char *desc, int3 *a);
void conf_lookup_vfloat_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3_ns(const Config *c, const char *ns, const char *desc, float3 *a);
void conf_lookup_vstring_ns(const Config *c, const char *ns, const char *d, int maxn, int *n, const char **ss);
// end::ns[]

// tag::opt[]
bool conf_opt_int(const Config *c, const char *desc, int *a);
bool conf_opt_float(const Config *c, const char *desc, float *a);
bool conf_opt_bool(const Config *c, const char *desc, int *a);
bool conf_opt_string(const Config *c, const char *desc, const char **a);
bool conf_opt_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
bool conf_opt_int3(const Config *c, const char *desc, int3 *a);
bool conf_opt_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
bool conf_opt_float3(const Config *c, const char *desc, float3 *a);
bool conf_opt_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);
// end::opt[]

// tag::set[]
void conf_set_int(const char *desc, int a, Config *c);
void conf_set_vint(const char *desc, int nelem, const int a[], Config *cfg);
void conf_set_int3(const char *desc, int3 a, Config *cfg);
void conf_set_float(const char *desc, float a, Config *cfg);
void conf_set_vfloat(const char *desc, int nelem, const float a[], Config *cfg);
void conf_set_float3(const char *desc, float3 a, Config *cfg);
void conf_set_bool(const char *desc, int a, Config *cfg);
void conf_set_string(const char *desc, const char *a, Config *cfg);
// end::set[]

// tag::dump[]
void conf_write_exe(const Config *cfg, FILE *stream);     // <1>
void conf_write_history(const Config *cfg, FILE *stream); // <2>
// end::dump[]
/* end src/conf/imp.h */

/* begin src/rbc/params/imp.h */
struct Config;
struct RbcParams;
struct RbcParams_v {
    float gammaC, gammaT, kBT;
    float kb, phi;
    float ks, x0, mpow;
    float ka, kd, kv;
    float totArea, totVolume;
};

// tag::mem[]
void rbc_params_ini(RbcParams **);
void rbc_params_fin(RbcParams *);
// end::mem[]

// tag::set[]
void rbc_params_set_fluct(float gammaC, float gammaT, float kBT, RbcParams *);
void rbc_params_set_bending(float kb, float phi, RbcParams *);
void rbc_params_set_spring(float ks, float x0, float mpow , RbcParams *);
void rbc_params_set_area_volume(float ka, float kd, float kv, RbcParams *);
void rbc_params_set_tot_area_volume(float totArea, float totVolume, RbcParams *);
// end::set[]

// tag::get[]
double rbc_params_get_tot_volume(const RbcParams *);
// end::get[]

// tag::cnf[]
void rbc_params_set_conf(const Config *c, const char *name, RbcParams *);
// end::cnf[]

// tag::view[]
RbcParams_v rbc_params_get_view(const RbcParams *);
// end::view[]
/* end src/rbc/params/imp.h */

void rbc_params_set_conf(const Config *c, const char *name, RbcParams *par) {
    float gc, gt, kbt, kb, phi, ks, mpow, x0, ka, kd, kv, totArea, totVolume;

    UC(conf_lookup_float_ns(c, name, "gammaC", &gc));
    UC(conf_lookup_float_ns(c, name, "gammaT", &gt));
    UC(conf_lookup_float_ns(c, name, "kBT", &kbt));
    UC(conf_lookup_float_ns(c, name, "kb", &kb));
    UC(conf_lookup_float_ns(c, name, "phi", &phi));
    UC(conf_lookup_float_ns(c, name, "ks", &ks));
    UC(conf_lookup_float_ns(c, name, "x0", &x0));
    UC(conf_lookup_float_ns(c, name, "mpow", &mpow));
    UC(conf_lookup_float_ns(c, name, "ka", &ka));
    UC(conf_lookup_float_ns(c, name, "kd", &kd));
    UC(conf_lookup_float_ns(c, name, "kv", &kv));

    UC(conf_lookup_float_ns(c, name, "totArea",   &totArea));
    UC(conf_lookup_float_ns(c, name, "totVolume", &totVolume));

    rbc_params_set_fluct(gc, gt, kbt, /**/ par);
    rbc_params_set_bending(kb, phi, /**/ par);
    rbc_params_set_spring(ks, x0, mpow, /**/ par);
    rbc_params_set_area_volume(ka, kd, kv, /**/ par);
    rbc_params_set_tot_area_volume(totArea, totVolume, /**/ par);
}
/* end src/rbc/params/conf.cpp */

#line 1 "src/rbc/params/imp.cpp"


/* begin src/rbc/params/imp.cpp */
#include <stdlib.h>
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

/* begin src/rbc/params/type.h */
/* model parameters */
struct RbcParams {
    float gammaC; /*  */
    float gammaT; /*  */
    float kBT;    /* temperature */
    float kb;     /* bending */
    float phi;    /* spontaneous angle */
    float ks;     /* spring constant */
    float x0;     /* spring extension relative to max. extension */
    float mpow;   /* exponent in POW spring */
    float ka;     /* global area constant */
    float kd;     /* local area constant */
    float kv;     /* volume constant */

    /* equilibrium */
    float totVolume; /* total volume */
    float totArea;   /* total area */
};
/* end src/rbc/params/type.h */


void rbc_params_ini(RbcParams **p) {
    EMALLOC(1, p);
}

void rbc_params_fin(RbcParams *p) {
    EFREE(p);
}

void rbc_params_set_fluct(float gammaC, float gammaT, float kBT, RbcParams *par) {
    par->gammaC = gammaC;
    par->gammaT = gammaT;
    par->kBT = kBT;
}

void rbc_params_set_bending(float kb, float phi, RbcParams *par) {
    par->kb = kb;
    par->phi = phi;
}

void rbc_params_set_spring(float ks, float x0, float mpow, RbcParams *par) {
    par->ks = ks;
    par->x0 = x0;
    par->mpow = mpow;
}

void rbc_params_set_tot_area_volume(float totArea, float totVolume, RbcParams *par) {
    par->totArea = totArea;
    par->totVolume = totVolume;
}

void rbc_params_set_area_volume(float ka, float kd, float kv, RbcParams *par) {
    par->ka = ka;
    par->kv = kv;
    par->kd = kd;
}

double rbc_params_get_tot_volume(const RbcParams *par) {
    return par->totVolume;
}

RbcParams_v rbc_params_get_view(const RbcParams *p) {
    RbcParams_v v;
    v.gammaC = p->gammaC;
    v.gammaT = p->gammaT;
    v.kBT = p->kBT;
    v.kb = p->kb;
    v.phi = p->phi;
    v.ks = p->ks;
    v.x0 = p->x0;
    v.mpow = p->mpow;
    v.ka = p->ka;
    v.kd = p->kd;
    v.kv = p->kv;
    v.totArea = p->totArea;
    v.totVolume = p->totVolume;
    return v;
}
/* end src/rbc/params/imp.cpp */

#line 1 "src/rbc/shape/imp.cpp"
