#include "macros.cuh"
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include <stdio.h>
#include "constants.cuh"
#include "types.cuh"
#include "device.cuh"

namespace cc { void check(int code, const char *file, int line); }
void emalloc(size_t, void **);
void efree(void *);
void error_stack_push(const char *file, int line);
void error_stack_pop();
bool error_get();
void error_report();
void error_abort();
void error_signal(const char *file, int line, const char *fmt, ...);
#define CC(ans) do { cc::check((ans), __FILE__, __LINE__); } while (0)
#define ERR(fmt, ...) do { error_signal(__FILE__, __LINE__, (fmt), ##__VA_ARGS__); error_report(); error_abort(); } while (0)
#define UC(F) do { error_stack_push(__FILE__, __LINE__); F; if (error_get()) { error_report(); error_abort(); } error_stack_pop(); } while (0)
#define EMALLOC(n, ppdata) UC(emalloc((n)*sizeof(**(ppdata)), (void**)(ppdata)))
#define EFREE(pdata) UC(efree(pdata))
void msg_print(const char *fmt, ...);
bool same_str(const char *, const char *);
struct Coords;
int xdomain(const Coords *); int ydomain(const Coords *); int zdomain(const Coords *);
float xl2xg(const Coords *, float); float yl2yg(const Coords *, float); float zl2zg(const Coords *, float);
enum {
    COLOR_NONE,
    COLOR_DROP,
    COLOR_UNIF
};

struct GenColor {
    int kind;
    float R;
};
static void set_color_unif(int n,      int* cc) {
    int i;
    for (i = 0; i < n; i++) cc[i] = BLUE_COLOR;
}
static void set_color_drop(float rad, const Coords *coords, int n, const Particle *pp,      int *cc) {
    enum {X, Y, Z};
    int lx, ly, lz;
    int i, r, g;
    bool inside;
    float x, y, z;
    float x0, y0, z0;
    Particle p;

    lx = xdomain(coords);
    ly = ydomain(coords);
    lz = zdomain(coords);

    x0 = 0.5*lx; y0 = 0.5*ly; z0 = 0.5*lz;
    enum {R = RED_COLOR, G = BLUE_COLOR};
    for (i = r = g = 0; i < n; i++) {
        p = pp[i];
        x = p.r[X]; y = p.r[Y]; z = p.r[Z];
        x = xl2xg(coords, x);
        y = yl2yg(coords, y);
        z = zl2zg(coords, z);
        x -= x0; y -= y0; z -= z0;
        inside = x*x + y*y + z*z < rad*rad;
        if (inside) {cc[i] = R; r++;}
        else        {cc[i] = G; g++;}
    }
    msg_print("color scheme: Bangladesh");
    msg_print("red/green : %d/%d", r, g);
}
void inter_color_ini(GenColor **c) {
    GenColor *gc;
    EMALLOC(1, c);
    gc = *c;
    gc->kind = COLOR_NONE;
}

void inter_color_fin(GenColor *c) {
    EFREE(c);
}

void inter_color_set_drop(float R, GenColor *c) {
    c->kind = COLOR_DROP;
    c->R = R;
}

void inter_color_set_uniform(GenColor *c) {
    c->kind = COLOR_UNIF;
}

void inter_color_apply_hst(const Coords *coords, const GenColor *gc, int n, const Particle *pp,      int *cc) {
    switch (gc->kind) {
    case COLOR_UNIF:
        set_color_unif(n,      cc);
        break;
    case COLOR_DROP:
        set_color_drop(gc->R, coords, n, pp,      cc);
        break;
    case COLOR_NONE:
        break;
    default:
        ERR("Unrecognised kind <%d>", gc->kind);
        break;
    };
}

void inter_color_apply_dev(const Coords *coords, const GenColor *gc, int n, const Particle *pp,      int *cc) {
    int *cc_hst;
    Particle *pp_hst;

    if (gc->kind == COLOR_NONE) return;

    EMALLOC(n, &cc_hst);
    EMALLOC(n, &pp_hst);

    cD2H(pp_hst, pp, n);

    UC(inter_color_apply_hst(coords, gc, n, pp_hst,      cc_hst));

    cH2D(cc, cc_hst, n);

    EFREE(pp_hst);
    EFREE(cc_hst);
}

struct Config;
void conf_lookup_string(const Config *, const char *, const char **);
void conf_lookup_float(const Config *, const char *, float *);
void inter_color_set_conf(const Config *cfg, GenColor *gc) {
    const char *kind;
    UC(conf_lookup_string(cfg, "iccolor.kind", &kind));
    if (same_str(kind, "uniform")) inter_color_set_uniform(gc);
    else if (same_str(kind, "drop")) {
        float R;
        UC(conf_lookup_float(cfg, "iccolor.R", &R));
        inter_color_set_drop(R, gc);
    } else if (!same_str(kind, "none")) ERR("unrecognised kind <%s>", kind);
}







