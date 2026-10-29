#define DEV_CUDA
#define CC_RELEASE
#include "macros.cuh"
#include "config.cuh"
#include "types.cuh"
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <vector_types.h>

struct Coords;
void local2global(const Coords *, float3, float3 *);
void error_stack_push(const char *, int);
void error_stack_pop();
bool error_get();
void error_report();
void error_abort();
void error_signal(const char *, int, const char *, ...);
void emalloc(size_t, void **);
void efree(void *);
void os_mkdir(const char *);
namespace m { bool is_master(MPI_Comm); }

struct IoBop { BopData *ff; BopData *ii; };
enum {NVARP = 6, NVARF = 3, NVARS = 6, NVARI = 1, NVARC = 1};
#define VARP "x y z vx vy vz"
#define VARF "fx fy fz"
#define VARS "sxx sxy sxz syy syz szz"
#define VARI "ids"
#define VARC "color"
#define PATTERN "%s-%05d"

void io_bop_ini(MPI_Comm comm, int maxp, IoBop **ib) {
    IoBop *t;
    EMALLOC(1, ib); t = *ib;
    BPC(bop_ini(&t->ff)); BPC(bop_ini(&t->ii));
    BPC(bop_set_n(maxp, t->ff)); BPC(bop_set_n(maxp, t->ii));
    BPC(bop_set_vars(NVARP + NVARF, VARP " " VARF, t->ff));
    BPC(bop_set_vars(NVARI, VARI, t->ii));
    BPC(bop_set_type(BopFLOAT, t->ff)); BPC(bop_set_type(BopINT, t->ii));
    BPC(bop_alloc(t->ff)); BPC(bop_alloc(t->ii));
    if (m::is_master(comm)) UC(os_mkdir(DUMP_BASE "/bop"));
}

void io_bop_fin(IoBop *t) {
    BPC(bop_fin(t->ff)); BPC(bop_fin(t->ii)); EFREE(t);
}

static void p2f3(const Particle p, float3 *r, float3 *v) {
    r->x = p.r[0]; r->y = p.r[1]; r->z = p.r[2];
    v->x = p.v[0]; v->y = p.v[1]; v->z = p.v[2];
}
static void f2f3(const Force fo, float3 *f) {
    f->x = fo.f[0]; f->y = fo.f[1]; f->z = fo.f[2];
}
static void copy_shift(const Coords *c, long n, const Particle *pp, float3 *w) {
    for (int j = 0; j < n; ++j) {
        float3 r, v; p2f3(pp[j], &r, &v); local2global(c, r, &w[2*j]); w[2*j + 1] = v;
    }
}
static void copy_shift_with_forces(const Coords *c, long n, const Particle *pp,
                                   const Force *ff, float3 *w) {
    for (int j = 0; j < n; ++j) {
        float3 r, v, f; p2f3(pp[j], &r, &v); f2f3(ff[j], &f);
        local2global(c, r, &w[3*j]); w[3*j + 1] = v; w[3*j + 2] = f;
    }
}
static void set_name(const char *base, int id, char *name) {
    sprintf(name, DUMP_BASE "/bop/" PATTERN, base, id);
}
static void write_bop(MPI_Comm cart, const char *base, int id, BopData *bop) {
    char fname[FILENAME_MAX]; set_name(base, id, fname);
    BPC(bop_write_header(cart, fname, bop)); BPC(bop_write_values(cart, fname, bop));
}

void io_bop_parts(MPI_Comm cart, const Coords *coords, long n, const Particle *pp,
                  const char *name, int id, IoBop *t) {
    BopData *bop = t->ff; float3 *out = (float3 *)bop_get_data(bop);
    copy_shift(coords, n, pp, out); BPC(bop_set_n(n, bop)); BPC(bop_set_vars(NVARP, VARP, bop));
    write_bop(cart, name, id, bop);
}
void io_bop_parts_forces(MPI_Comm cart, const Coords *coords, long n, const Particle *pp,
                         const Force *ff, const char *name, int id, IoBop *t) {
    BopData *bop = t->ff; float3 *out = (float3 *)bop_get_data(bop);
    copy_shift_with_forces(coords, n, pp, ff, out); BPC(bop_set_n(n, bop));
    BPC(bop_set_vars(NVARP + NVARF, VARP " " VARF, bop)); write_bop(cart, name, id, bop);
}
void io_bop_stresses(MPI_Comm cart, long n, const float *ss, const char *name, int id, IoBop *t) {
    BopData *bop = t->ff; float *out = (float *)bop_get_data(bop);
    memcpy(out, ss, n * NVARS * sizeof(float)); BPC(bop_set_n(n, bop));
    BPC(bop_set_vars(NVARS, VARS, bop)); write_bop(cart, name, id, bop);
}
void io_bop_ids(MPI_Comm cart, long n, const int *ii, const char *name, int id, IoBop *t) {
    BopData *bop = t->ii; int *out = (int *)bop_get_data(bop);
    memcpy(out, ii, n * NVARI * sizeof(int)); BPC(bop_set_n(n, bop));
    BPC(bop_set_vars(NVARI, VARI, bop)); write_bop(cart, name, id, bop);
}
void io_bop_colors(MPI_Comm cart, long n, const int *cc, const char *name, int id, IoBop *t) {
    BopData *bop = t->ii; int *out = (int *)bop_get_data(bop);
    memcpy(out, cc, n * NVARC * sizeof(int)); BPC(bop_set_n(n, bop));
    BPC(bop_set_vars(NVARC, VARC, bop)); write_bop(cart, name, id, bop);
}

#undef VARP
#undef VARF
#undef VARS
#undef VARI
#undef VARC
#undef PATTERN
