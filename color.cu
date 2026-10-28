#include "macros.cuh"
#define DEV_CUDA
#define KL_RELEASE
#include "device.cuh"
#include "launch.cuh"
#include "types.cuh"
#include <assert.h>
#include <curand_kernel.h>

namespace cc { void check(int, const char*, int); }
#define CC(ans) do { cc::check((ans), __FILE__, __LINE__); } while (0)

struct Coords;
struct Coords_v { int xc, yc, zc, xd, yd, zd, Lx, Ly, Lz; };
void coords_get_view(const Coords*, Coords_v*);
int xlo(const Coords*); int ylo(const Coords*); int zlo(const Coords*);
int xhi(const Coords*); int yhi(const Coords*); int zhi(const Coords*);
bool is_end(const Coords*, int);

struct Rnd {
    curandState_t * state;
} color_rnd;
__global__ void color_linear_flux_kernel(int3 L, int dir, int color, int n, const Particle *pp, int *cc) {
    int i;
    Particle p;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    p = pp[i];
    const int HL[] = {L.x/2, L.y/2, L.z/2};

    if (p.r[dir] >= HL[dir])
        cc[i] = color;
}

__global__ void color_radial_flux_kernel(const Coords_v c, const float R, const float Po, int color, int n, const Particle *pp, int *cc, curandState_t *rg) {

    long i = threadIdx.x + blockIdx.x * blockDim.x;
    curandState_t rnd_state = rg[i];
    double rand = curand_normal(&rnd_state);
    double cdf = normcdf(rand);

    if (cdf <= Po)
    {
        int i;
        Particle p;
        i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n) return;
        p = pp[i];

        float pxc = p.r[0] - 0.5f * c.Lx * (c.xd - 2 * c.xc - 1);
        float pyc = p.r[1] - 0.5f * c.Ly * (c.yd - 2 * c.yc - 1);
        float pzc = p.r[2] - 0.5f * c.Lz * (c.zd - 2 * c.zc - 1);

        const float r = sqrtf( pxc*pxc + pyc*pyc + pzc*pzc );

        if (r <= R) {
            cc[i] = color;
        }
    }
}

__global__ void color_decolor_tracer_kernel(const Coords_v c, const float R, const float Po, const int color, const int n, const Particle *pp, int *cc, curandState_t *rg) {

    int i;
    i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= n)
        return;

    if (cc[i] == color)
        return;

    Particle p;
    p = pp[i];

    curandState_t rnd_state = rg[i];
    double rand = curand_normal(&rnd_state);
    double cdf = normcdf(rand);

    if (cdf <= Po)
    {
        float pxc = p.r[0] - 0.5f * c.Lx * (c.xd - 2 * c.xc - 1);
        float pyc = p.r[1] - 0.5f * c.Ly * (c.yd - 2 * c.yc - 1);
        float pzc = p.r[2] - 0.5f * c.Lz * (c.zd - 2 * c.zc - 1);

        const float r = sqrtf( pxc*pxc + pyc*pyc + pzc*pzc );

        if (r > R)
            cc[i] = color;
    }
}

__global__ void color_ini_rnd_kernel(long seed, int n, curandState_t *state) {
    long i;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    curand_init(seed, i, 0, &state[i]);
}

void color_linear_flux(const Coords *coords, int3 L, int dir, int color, int n, const Particle *pp, int *cc) {
    assert(dir >= 0 && dir <= 2);
    if (is_end(coords, dir))
        KL(color_linear_flux_kernel, (k_cnf(n)), (L, dir, color, n, pp, cc));
}

void color_tracers(const Coords *coords, int color, const float R, const float Po, int n, const Particle *pp, int *cc) {

    int l[3] = {xlo(coords), ylo(coords), zlo(coords)};
    int h[3] = {xhi(coords), yhi(coords), zhi(coords)};


    bool inradius = 1;
    for (int i=0; i<2; ++i)
        inradius *= ( sqrtf(l[i]*l[i]) <= R || sqrtf(h[i]*h[i]) <= R );

    if (inradius) {
        Coords_v view;
        coords_get_view(coords, &view);
    KL(color_radial_flux_kernel, (k_cnf(n)), (view, R, Po, color, n, pp, cc, color_rnd.state));
    }
}

void decolor_tracers(const Coords *coords, int color, const float R, const float Po, int n, const Particle *pp, int *cc) {
    Coords_v view;
    coords_get_view(coords, &view);
    KL(color_decolor_tracer_kernel, (k_cnf(n)), (view, R, Po, color, n, pp, cc, color_rnd.state));
}

static void color_ini_rnd(int n, curandState_t *state) {
    long seed = 1234567;
    KL(color_ini_rnd_kernel, (k_cnf(n)), (seed, n, state));
}

void tracers_ini(const int n) {
    Dalloc(&color_rnd.state, n);
    color_ini_rnd(n, color_rnd.state);
}

void tracers_fin() {
    Dfree(color_rnd.state);
}
