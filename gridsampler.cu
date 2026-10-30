#include "memory.cuh"
#include "macros.cuh"
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include <mpi.h>
#include <stdio.h>
#include "config.cuh"
#include "constants.cuh"
#include "types.cuh"
#include "device.cuh"
#include "launch.cuh"

namespace cc { void check(int code, const char *file, int line); }
void error_stack_push(const char *file, int line);
void error_stack_pop();
bool error_get();
void error_report();
void error_abort();
void error_signal(const char *file, int line, const char *fmt, ...);
#define CC(ans) do { cc::check((ans), __FILE__, __LINE__); } while (0)
struct GridSampler;
struct GridSampleData;
void grid_sampler_reset(GridSampler *);
void grid_sampler_data_reset(GridSampleData *);
void grid_write(int3, int3, MPI_Comm, const char *, int, const float **, const char **);
enum {
    RHO, VX, VY, VZ,
    NFIELDS_P
};

enum {
    SXX, SXY, SXZ,
    SYY, SYZ, SZZ,
    NFIELDS_S
};

enum {
    NFIELDS_C = N_COLOR
};

enum {
    TOT_NFIELDS = NFIELDS_P + NFIELDS_S + NFIELDS_C
};

static const char *names_p[NFIELDS_P] =
    {"density", "u", "v", "w"};

static const char *names_s[NFIELDS_S] =
    {"sxx", "sxy", "sxz",
     "syy", "syz", "szz"};

#define make_str(a) #a ,
static const char *names_c[NFIELDS_C] =
    {XMACRO_COLOR(make_str)};
#undef make_str

struct Grid {
    int3 L, N;
    float *p[NFIELDS_P];
    float *s[NFIELDS_S];
    float *c[NFIELDS_C];
    bool colors, stress;
};

struct GridSampler {
    Grid sdev, stdev, hst;
    long nsteps;
};

struct SampleDatum {
    long n;
    const Particle *pp;
    const float *ss;
    const int *cc;
};

enum {
    MAX_N_DATA = 10
};

struct GridSampleData {
    int n;
    SampleDatum d[MAX_N_DATA];
};

struct SamplerPart {
    float3 r, v;
};

struct SamplerStress {
    float3 s1, s2;
};
#define _S_ static __device__
#define _I_ static __device__

_I_ void sampler_fetch_part(int i, const SampleDatum *d, SamplerPart *p) {
    float2 p0, p1, p2;
    const float2 *pp = (const float2*) d->pp;
    p0 = pp[3*i+0]; p1 = pp[3*i+1]; p2 = pp[3*i+2];

    p->r = make_float3(p0.x, p0.y, p1.x);
    p->v = make_float3(p1.y, p2.x, p2.y);
}

_I_ void sampler_fetch_color(int i, const SampleDatum *d, int *c) {
    *c = d->cc[i];
}

_I_ void sampler_fetch_stress(int i, const SampleDatum *d, SamplerStress *s) {
    float2 s0, s1, s2;
    const float2 *ss = (const float2*) d->ss;
    s0 = ss[3*i+0]; s1 = ss[3*i+1]; s2 = ss[3*i+2];

    s->s1 = make_float3(s0.x, s0.y, s1.x);
    s->s2 = make_float3(s1.y, s2.x, s2.y);
}

#undef _S_
#undef _I_
#define _S_ static __device__
#define _I_ static __device__

_I_ void sampler_add_part(int i, const SamplerPart *p, const Grid *g) {
    atomicAdd(g->p[RHO] + i, 1.f);
    atomicAdd(g->p[VX]  + i, p->v.x);
    atomicAdd(g->p[VY]  + i, p->v.y);
    atomicAdd(g->p[VZ]  + i, p->v.z);
}

_I_ void sampler_add_color(int i, const int c, const Grid *g) {
#define ADD_COLOR(a)                               \
    if (ENUM_COLOR(a) == c)                        \
        atomicAdd(g->c[ENUM_COLOR(a)] + i, 1.f);

    XMACRO_COLOR(ADD_COLOR)

#undef ADD_COLOR
}

_I_ void sampler_add_stress(int i, const SamplerStress *s, const Grid *g) {
    atomicAdd(g->s[SXX] + i, s->s1.x);
    atomicAdd(g->s[SXY] + i, s->s1.y);
    atomicAdd(g->s[SXZ] + i, s->s1.z);

    atomicAdd(g->s[SYY] + i, s->s2.x);
    atomicAdd(g->s[SYZ] + i, s->s2.y);
    atomicAdd(g->s[SZZ] + i, s->s2.z);
}

#undef _S_
#undef _I_
#define _S_ static __device__

_S_ float sampler_get_spacing(int L, int N) {
    return (float) L / (float) N;
}

_S_ int sampler_get_coord(float x, int L, int N) {
    int i;
    float dx;
    dx = sampler_get_spacing(L, N);
    i = (x + L/2) / dx;
    return min(N-1, max(0, i));
}

_S_ int3 sampler_get_cell_coords(float3 r, int3 L, int3 N) {
    return make_int3(sampler_get_coord(r.x, L.x, N.x),
                     sampler_get_coord(r.y, L.y, N.y),
                     sampler_get_coord(r.z, L.z, N.z));
}

_S_ int sampler_get_grid_id(int3 gc, int3 N) {
    return gc.x + N.x * (gc.y + N.y * gc.z);
}

_S_ int sampler_get_grid_id(const SamplerPart *p, const Grid *g) {
    int3 gcoords;
    gcoords = sampler_get_cell_coords(p->r, g->L, g->N);
    return sampler_get_grid_id(gcoords, g->N);
}

__global__ void sampler_add_kernel(SampleDatum data,      Grid grid) {
    int pid, gid;
    SamplerPart p;
    pid = threadIdx.x + blockIdx.x * blockDim.x;
    if (pid >= data.n) return;

    sampler_fetch_part(pid, &data, &p);
    gid = sampler_get_grid_id(&p, &grid);

    sampler_add_part(gid, &p, &grid);

    if (grid.colors) {
        int c;
        sampler_fetch_color(pid, &data, &c);
        sampler_add_color(gid, c, &grid);
    }

    if (grid.stress) {
        SamplerStress s;
        sampler_fetch_stress(pid, &data, &s);
        sampler_add_stress(gid, &s, &grid);
    }
}

_S_ long sampler_get_size(const Grid *g) {
    int3 N = g->N;
    return N.x * N.y * N.z;
}

_S_ float sampler_cell_volume(const Grid *g) {
    float dx, dy, dz;
    dx = sampler_get_spacing(g->L.x, g->N.x);
    dy = sampler_get_spacing(g->L.y, g->N.y);
    dz = sampler_get_spacing(g->L.z, g->N.z);
    return dx * dy * dz;
}

__global__ void sampler_space_avg_kernel(Grid g) {
    long i;
    float inv_rho;
    float rho, vol;
    float sv, sr, ss;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= sampler_get_size(&g)) return;

    rho = g.p[RHO][i];
    vol = sampler_cell_volume(&g);

    inv_rho = fabs(rho) > 1e-6 ? 1.f / rho : 0;

    sv = inv_rho;
    sr = 1.f / vol;

    g.p[RHO][i] = sr * rho;

    g.p[VX][i] *= sv;
    g.p[VY][i] *= sv;
    g.p[VZ][i] *= sv;

    if (g.colors) {
#define scale(a) g.c[ENUM_COLOR(a)][i] *= inv_rho;
        XMACRO_COLOR(scale)
#undef scale
    }

    if (g.stress) {
        ss = sr;
        g.s[SXX][i] *= ss;
        g.s[SXY][i] *= ss;
        g.s[SXZ][i] *= ss;
        g.s[SYY][i] *= ss;
        g.s[SYZ][i] *= ss;
        g.s[SZZ][i] *= ss;
    }
}

__global__ void sampler_add_to_grid_kernel(const Grid src, Grid dst) {
    long i, j;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= sampler_get_size(&src)) return;

    for (j = 0; j < NFIELDS_P; ++j)
        dst.p[j][i] += src.p[j][i];

    if (src.colors) {
        for (j = 0; j < NFIELDS_C; ++j)
            dst.c[j][i] += src.c[j][i];
    }

    if (src.stress) {
        for (j = 0; j < NFIELDS_S; ++j)
            dst.s[j][i] += src.s[j][i];
    }
}

__global__ void sampler_time_avg_kernel(int nsteps, Grid g) {
    long i, j;
    float s;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= sampler_get_size(&g)) return;

    s = 1.0 / nsteps;

    for (j = 0; j < NFIELDS_P; ++j)
        g.p[j][i] *= s;

    if (g.colors)
        for (j = 0; j < NFIELDS_C; ++j)
            g.c[j][i] *= s;

    if (g.stress)
        for (j = 0; j < NFIELDS_S; ++j)
            g.s[j][i] *= s;
}

#undef _S_

static long get_size(const Grid *g) {
    int3 N = g->N;
    return N.x * N.y * N.z;
}

static void dev_alloc_n(int na, long n, float *a[]) {
    for (int i = 0; i < na; ++i) Dalloc(&a[i], n);
}

static void ini_dev_grid(bool colors, bool stress, int3 L, int3 N, Grid *g) {
    long n;
    g->N = N;
    g->L = L;
    g->colors = colors;
    g->stress = stress;

    n = get_size(g);

    UC(             dev_alloc_n(NFIELDS_P, n, g->p) );
    if (colors) UC( dev_alloc_n(NFIELDS_C, n, g->c) );
    if (stress) UC( dev_alloc_n(NFIELDS_S, n, g->s) );
}

static void hst_alloc_n(int na, long n, float *a[]) {
    for (int i = 0; i < na; ++i) EMALLOC(n, &a[i]);
}

static void ini_hst_grid(bool colors, bool stress, int3 L, int3 N, Grid *g) {
    long n;
    g->N = N;
    g->L = L;
    g->colors = colors;
    g->stress = stress;

    n = get_size(g);
    UC(             hst_alloc_n(NFIELDS_P, n, g->p) );
    if (colors) UC( hst_alloc_n(NFIELDS_C, n, g->c) );
    if (stress) UC( hst_alloc_n(NFIELDS_S, n, g->s) );
}

static void dev_free_n(int na, float *a[]) {
    for (int i = 0; i < na; ++i) Dfree(a[i]);
}

static void fin_dev_grid(Grid *g) {
    UC(                dev_free_n(NFIELDS_P, g->p) );
    if (g->colors) UC( dev_free_n(NFIELDS_C, g->c) );
    if (g->stress) UC( dev_free_n(NFIELDS_S, g->s) );
}

static void hst_free_n(int na, float *a[]) {
    for (int i = 0; i < na; ++i) EFREE(a[i]);
}

static void fin_hst_grid(Grid *g) {
    UC(                hst_free_n(NFIELDS_P, g->p) );
    if (g->colors) UC( hst_free_n(NFIELDS_C, g->c) );
    if (g->stress) UC( hst_free_n(NFIELDS_S, g->s) );
}

void grid_sampler_ini(bool colors, bool stress, int3 L, int3 N, GridSampler **s0) {
    GridSampler *s;
    EMALLOC(1, s0);
    s = *s0;
    UC(ini_dev_grid(colors, stress, L, N, &s->sdev));
    UC(ini_dev_grid(colors, stress, L, N, &s->stdev));
    UC(ini_hst_grid(colors, stress, L, N, &s->hst));
    UC(grid_sampler_reset(s));
}

void grid_sampler_fin(GridSampler *s) {
    UC(fin_dev_grid(&s->sdev));
    UC(fin_dev_grid(&s->stdev));
    UC(fin_hst_grid(&s->hst));
    EFREE(s);
}

static void reset_dev_n(int na, long n, float *a[]) {
    for (int i = 0; i < na; ++i) DzeroA(a[i], n);
}

static void reset_dev_grid(Grid *g) {
    long n = get_size(g);
    UC(                reset_dev_n(NFIELDS_P, n, g->p) );
    if (g->colors) UC( reset_dev_n(NFIELDS_C, n, g->c) );
    if (g->stress) UC( reset_dev_n(NFIELDS_S, n, g->s) );
}

void grid_sampler_reset(GridSampler *s) {
    s->nsteps = 0;
    UC(reset_dev_grid(&s->stdev));
}

static void add_data_to_grid(const GridSampleData *data, Grid *g) {
    long i, n;
    SampleDatum d;
    for (i = 0; i < data->n; ++i) {
        d = data->d[i];
        n = d.n;
        KL(sampler_add_kernel, (k_cnf(n)), (d, *g));
    }
}

static void space_avg(Grid *g) {
    long n = get_size(g);
    KL(sampler_space_avg_kernel, (k_cnf(n)), (*g));
}

static void add_to_grid(const Grid *s, Grid *st) {
    long n = get_size(s);
    KL(sampler_add_to_grid_kernel, (k_cnf(n)), (*s, *st));
}

void grid_sampler_add(const GridSampleData *data, GridSampler *s) {
    Grid *sg, *stg;
    sg  = &s->sdev;
    stg = &s->stdev;
    UC(reset_dev_grid(sg));
    UC(add_data_to_grid(data, sg));
    UC(space_avg(sg));
    UC(add_to_grid(sg, stg));
    s->nsteps ++;
}

static void time_avg(int nsteps, Grid *g) {
    long n = get_size(g);
    KL(sampler_time_avg_kernel, (k_cnf(n)), (nsteps, *g));
}

static void download_n(int na, long n, float * const dev[], float *hst[]) {
   for (int i = 0; i < na; ++i) aD2H(hst[i], dev[i], n);
}

static void download(const Grid *dev, Grid *hst) {
    long n = get_size(dev);
    UC(                  download_n(NFIELDS_P, n, dev->p, hst->p) );
    if (dev->colors) UC( download_n(NFIELDS_C, n, dev->c, hst->c) );
    if (dev->stress) UC( download_n(NFIELDS_S, n, dev->s, hst->s) );
}

static void build_desc(int ncmp, const char * const name[], const float * const data[],
                       int *j, const char *dst_name[], const float *dst_data[]) {
    for (int i = 0; i < ncmp; ++i) {
        dst_data[*j] = data[i];
        dst_name[*j] = name[i];
        ++(*j);
    }
}

static void dump(MPI_Comm cart, const char *dir, long id, const Grid *g) {
    char path[FILENAME_MAX] = {'\0'};
    int ncmp;
    const float *data[TOT_NFIELDS];
    const char *names[TOT_NFIELDS];

    ncmp = 0;
    build_desc               (NFIELDS_P, names_p, g->p,      &ncmp, names, data);
    if (g->colors) build_desc(NFIELDS_C, names_c, g->c,      &ncmp, names, data);
    if (g->stress) build_desc(NFIELDS_S, names_s, g->s,      &ncmp, names, data);

    snprintf(path, FILENAME_MAX, DUMP_BASE "/%s/%04ld.raw", dir, id);

    UC(grid_write(g->N, g->L, cart, path, ncmp, data, names));
}

void grid_sampler_dump(MPI_Comm cart, const char *dir, long id, GridSampler *s) {
    UC(time_avg(s->nsteps, &s->stdev));
    UC(download(&s->stdev, &s->hst));
    dSync();
    UC(dump(cart, dir, id, &s->hst));
}
void grid_sampler_data_ini(GridSampleData **s) {
    EMALLOC(1, s);
    grid_sampler_data_reset(*s);
}

void grid_sampler_data_fin(GridSampleData  *s) {
    EFREE(s);
}

void grid_sampler_data_reset(GridSampleData *s) {
    s->n = 0;
}

void grid_sampler_data_push(long n, const Particle *pp, const int *cc, const float *ss, GridSampleData *s) {
    int id;
    SampleDatum *d;
    id = s->n++;

    if (s->n >= MAX_N_DATA)
        ERR("Too many data: %d/%d", s->n / MAX_N_DATA);

    d = s->d + id;
    d->n = n;
    d->pp = pp;
    d->cc = cc;
    d->ss = ss;
}

