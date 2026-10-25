// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/wall/wvel/conf.cpp */
#include <stdio.h>
#include <vector_types.h>

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

/* begin src/wall/wvel/imp.h */
struct Wvel;
struct WvelStep;

struct WvelCste_v;
struct WvelShear_v;
struct WvelHS_v;

struct float3;
struct Config;

#ifndef UDX_WALL_WVEL_ENUM
#define UDX_WALL_WVEL_ENUM
// tag::enum[]
enum {
    WALL_VEL_V_CSTE,
    WALL_VEL_V_SHEAR,
    WALL_VEL_V_HS,
};
// end::enum[]
#endif

// tag::mem[]
void wvel_ini(Wvel **wv);
void wvel_fin(Wvel *wv);

void wvel_step_ini(WvelStep **wv);
void wvel_step_fin(WvelStep *wv);
// end::mem[]

// tag::ini[]
void wvel_set_cste(float3 u, Wvel *vw);
void wvel_set_shear(float gdot, int vdir, int gdir, int half, Wvel *vw);
void wvel_set_shear_sin(float gdot, int vdir, int gdir, int half, float w, Wvel *vw);
void wvel_set_hs(float u, float h, Wvel *vw);
void wvel_set_timestep(float dt, Wvel *vw);
// end::ini[]

// tag::cnf[]
void wvel_set_conf(const Config *cfg, Wvel *vw);
// end::cnf[]

// tag::step[]
void wvel_get_step(float t, const Wvel *wv, /**/ WvelStep *);
// end::step[]

// tag::get[]
int wvel_get_type(const WvelStep *w);
// end::get[]

// tag::view[]
void wvel_get_view(const WvelStep *w, /**/ WvelCste_v *v);
void wvel_get_view(const WvelStep *w, /**/ WvelShear_v *v);
void wvel_get_view(const WvelStep *w, /**/ WvelHS_v *v);
// end::view[]
/* end src/wall/wvel/imp.h */

void wvel_set_conf(const Config *cfg, Wvel *vw) {
    const char *type;
    UC(conf_lookup_string(cfg, "wvel.type", &type));

    if (same_str(type, "constant")) {
        float3 u;
        UC(conf_lookup_float3(cfg, "wvel.u", &u));
        UC(wvel_set_cste(u, vw));
    }
    else if (same_str(type, "shear")) {
        float gdot;
        int vdir, gdir, half;
        UC(conf_lookup_float(cfg, "wvel.gdot", &gdot));
        UC(conf_lookup_int(cfg, "wvel.vdir", &vdir));
        UC(conf_lookup_int(cfg, "wvel.gdir", &gdir));
        UC(conf_lookup_int(cfg, "wvel.half", &half));
        UC(wvel_set_shear(gdot, vdir, gdir, half, vw));
    }
    else if (same_str(type, "shear sin")) {
        float gdot, w;
        int vdir, gdir, half;
        UC(conf_lookup_float(cfg, "wvel.gdot", &gdot));
        UC(conf_lookup_int(cfg, "wvel.vdir", &vdir));
        UC(conf_lookup_int(cfg, "wvel.gdir", &gdir));
        UC(conf_lookup_int(cfg, "wvel.half", &half));
        UC(conf_lookup_float(cfg, "wvel.w", &w));
        UC(wvel_set_shear_sin(gdot, vdir, gdir, half, w, vw));
    }
    else if (same_str(type, "hele shaw")) {
        float u, h;
        UC(conf_lookup_float(cfg, "wvel.u", &u));
        UC(conf_lookup_float(cfg, "wvel.h", &h));
        UC(wvel_set_hs(u, h, vw));
    }
    else {
        ERR("unknown type <%s>\n", type);
    }
}
/* end src/wall/wvel/conf.cpp */

#line 1 "src/wall/wvel/imp.cpp"


/* begin src/wall/wvel/imp.cpp */
#include <mpi.h>
#include <math.h>
#include <vector_types.h>
#include <assert.h>

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

/* begin src/utils/msg.h */
// tag::int[]
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
// end::int[]
/* end src/utils/msg.h */

/* begin src/coords/imp.h */
struct Coords;
struct Coords_v;
struct float3;
struct int3;

// tag::view[]
void coords_get_view(const Coords *c, Coords_v *v);
// end::view[]

// tag::domainsz[]
int xdomain(const Coords *c);
int ydomain(const Coords *c);
int zdomain(const Coords *c);
// end::domainsz[]

// tag::local2center[]
float xl2xc(const Coords *c, float xl);
float yl2yc(const Coords *c, float yl);
float zl2zc(const Coords *c, float zl);
void local2center(const Coords *c, float3 rl, /**/ float3 *rc);
// end::local2center[]

// tag::center2local[]
float xc2xl(const Coords *c, float xc);
float yc2yl(const Coords *c, float yc);
float zc2zl(const Coords *c, float zc);
void center2local(const Coords *c, float3 rc, /**/ float3 *rl);
// end::center2local[]

// tag::local2global[]
float xl2xg(const Coords *c, float xl);
float yl2yg(const Coords *c, float yl);
float zl2zg(const Coords *c, float zl);
void local2global(const Coords *c, float3 rl, /**/ float3 *rg);
// end::local2global[]

// tag::global2local[]
float xg2xl(const Coords *c, float xg);
float yg2yl(const Coords *c, float yg);
float zg2zl(const Coords *c, float zg);
void global2local(const Coords *c, float3 rg, /**/ float3 *rl);
// end::global2local[]

/* edges of the sub-domain in global coordinates */

// tag::bblocal[]
int xlo(const Coords*);
int ylo(const Coords*);
int zlo(const Coords*);

int xhi(const Coords*);
int yhi(const Coords*);
int zhi(const Coords*);
// end::bblocal[]

// tag::subdomainsz[]
int xs(const Coords*);
int ys(const Coords*);
int zs(const Coords*);
int3 subdomain(const Coords*);
// end::subdomainsz[]

// tag::int[]
/* rank predicates */
bool is_end(const Coords *c, int dir);

/* a string unique for a rank */
void coord_stamp(const Coords *c, /**/ char *s);

/* number of subdomains */
int coords_size(const Coords *c);
// end::int[]
/* end src/coords/imp.h */

#ifndef UDX_WALL_WVEL_TYPE
#define UDX_WALL_WVEL_TYPE
/* begin src/wall/wvel/type.h */
/* view structures                      */
/* (what is passed to device functions) */

struct WvelCste_v {
    float3 u;
};

struct WvelShear_v {
    float gdot;     // shear rate
    int vdir, gdir; // direction of the flow and gradient
    int half;       // only half is moving ?
};

struct WvelHS_v {
    float u; // radial max inflow
    float h; // height of the channel
};

/* end src/wall/wvel/type.h */
#endif


/* begin src/wall/wvel/imp/type.h */
/* data stored on the host                       */
/* (helpers to setup the views at each timestep) */

struct WvelCste {
    float3 u; // velocity amplitude
};

struct WvelShear {
    float gdot;     // shear rate
    int vdir, gdir; // direction of the flow and gradient
    int half;       // only half is moving ?
};

struct WvelShearSin : WvelShear {
    float w;        // frequency
};

struct WvelHS {
    float u; // radial max inflow
    float h; // height of the channel
};

enum {
    WALL_VEL_CSTE,
    WALL_VEL_SHEAR,
    WALL_VEL_SHEAR_SIN,
    WALL_VEL_HS,
};

union WvelPar {
    WvelCste cste;
    WvelShear shear;
    WvelShearSin shearsin;
    WvelHS hs;
};

/* main structure */

struct Wvel {
    WvelPar p;  /* parameters             */
    int type;
};

/* step */

union WvelPar_v {
    WvelCste_v cste;
    WvelShear_v shear;
    WvelHS_v hs;
};

/* device structure: to be passed to device code */

struct WvelStep {
    WvelPar_v p;
    int type;
};

/* end src/wall/wvel/imp/type.h */
/* begin src/wall/wvel/imp/main.h */
void wvel_ini(Wvel **wv) {
    EMALLOC(1, wv);
}

void wvel_fin(Wvel *wv) {
    EFREE(wv);
}

void wvel_step_ini(WvelStep **wv) {
    EMALLOC(1, wv);
}

void wvel_step_fin(WvelStep *wv) {
    EFREE(wv);
}

void wvel_set_cste(float3 u, Wvel *vw) {
    WvelCste p;
    p.u = u;
    vw->type = WALL_VEL_CSTE;
    vw->p.cste = p;
}

void wvel_set_shear(float gdot, int vdir, int gdir, int half, Wvel *vw) {
    WvelShear p;
    p.gdot = gdot;
    p.vdir = vdir;
    p.gdir = gdir;
    p.half = half;

    vw->type = WALL_VEL_SHEAR;
    vw->p.shear = p;
}

void wvel_set_shear_sin(float gdot, int vdir, int gdir, int half, float w, Wvel *vw) {
    WvelShearSin p;
    p.gdot = gdot;
    p.vdir = vdir;
    p.gdir = gdir;
    p.half = half;
    p.w    = w;

    vw->type = WALL_VEL_SHEAR_SIN;
    vw->p.shearsin = p;
}

void wvel_set_hs(float u, float h, Wvel *vw) {
    WvelHS p;
    p.u = u;
    p.h = h;
    vw->type = WALL_VEL_HS;
    vw->p.hs = p;
}

static void set_step_cste(WvelCste p, WvelStep *wv) {
    wv->type = WALL_VEL_V_CSTE;
    wv->p.cste.u = p.u;
}

static void set_step_shear(WvelShear p, WvelStep *wv) {
    wv->type = WALL_VEL_V_SHEAR;
    wv->p.shear.gdot = p.gdot;
    wv->p.shear.gdir = p.gdir;
    wv->p.shear.vdir = p.vdir;
    wv->p.shear.half = p.half;
}

static void set_step_shear(float t, WvelShearSin p, WvelStep *wv) {
    float gdot;
    float w;
    wv->type = WALL_VEL_V_SHEAR;
    w = p.w;
    gdot = p.gdot * sin(w * t);

    wv->p.shear.gdot = gdot;
    wv->p.shear.gdir = p.gdir;
    wv->p.shear.vdir = p.vdir;
    wv->p.shear.half = p.half;
}

static void set_step_hs(WvelHS p, WvelStep *wv) {
    wv->type = WALL_VEL_V_HS;
    wv->p.hs.u = p.u;
    wv->p.hs.h = p.h;
}

void wvel_get_step(float t, const Wvel *wv, /**/ WvelStep *view) {
    switch (wv->type) {
    case WALL_VEL_CSTE:
        set_step_cste(wv->p.cste, /**/ view);
        break;
    case WALL_VEL_SHEAR:
        set_step_shear(wv->p.shear, /**/ view);
        break;
    case WALL_VEL_SHEAR_SIN:
        set_step_shear(t, wv->p.shearsin, /**/ view);
        break;
    case WALL_VEL_HS:
        set_step_hs(wv->p.hs, /**/ view);
        break;
    default:
        ERR("wrong type provided: <%d>", wv->type);
        break;
    };
}

int wvel_get_type(const WvelStep *w) {
    return w->type;
}

void wvel_get_view(const WvelStep *w, /**/ WvelCste_v *v) {
    *v = w->p.cste;
}

void wvel_get_view(const WvelStep *w, /**/ WvelShear_v *v) {
    *v = w->p.shear;
}

void wvel_get_view(const WvelStep *w, /**/ WvelHS_v *v) {
    *v = w->p.hs;
}
/* end src/wall/wvel/imp/main.h */
/* end src/wall/wvel/imp.cpp */

#line 1 "src/main.cpp"
