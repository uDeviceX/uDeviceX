namespace color_dev
{

__global__ void linear_flux(int3 L, int dir, int color, int n, const Particle *pp, int *cc) {
    int i;
    Particle p;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    p = pp[i];
    const int HL[] = {L.x/2, L.y/2, L.z/2};

    if (p.r[dir] >= HL[dir])
        cc[i] = color;
}

__global__ void radial_flux(const Coords_v c, const float R, const float Po, int color, int n, const Particle *pp, int *cc, curandState_t *rg) {

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

__global__ void decolor_tracer(const Coords_v c, const float R, const float Po, const int color, const int n, const Particle *pp, int *cc, curandState_t *rg) {

    int i;
    i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= n)
        return;

    if (cc[i] == color) //if blue color nothing todo
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

        if (r > R) // if outside the initialization region
            cc[i] = color;
    }
}

__global__ void ini_rnd(long seed, int n, curandState_t *state) {
    long i;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    curand_init(seed, i, 0, &state[i]);
}

}


void color_linear_flux(const Coords *coords, int3 L, int dir, int color, int n, const Particle *pp, int *cc) {
    assert(dir >= 0 && dir <= 2);
    if (is_end(coords, dir))
        KL(color_dev::linear_flux, (k_cnf(n)), (L, dir, color, n, pp, cc));
}

void color_tracers(const Coords *coords, int color, const float R, const float Po, int n, const Particle *pp, int *cc) {
    // find min and max of my domain in global coordinate system
    int l[3] = {xlo(coords), ylo(coords), zlo(coords)};
    int h[3] = {xhi(coords), yhi(coords), zhi(coords)};

    // if center (0,0,0) is in my domain, proceed
    bool inradius = 1;
    for (int i=0; i<2; ++i)
        inradius *= ( sqrtf(l[i]*l[i]) <= R || sqrtf(h[i]*h[i]) <= R );

    if (inradius) {
        Coords_v view;
        coords_get_view(coords, &view);
        KL(color_dev::radial_flux, (k_cnf(n)), (view, R, Po, color, n, pp, cc, rnd.state));
    }
}

void decolor_tracers(const Coords *coords, int color, const float R, const float Po, int n, const Particle *pp, int *cc) {
    Coords_v view;
    coords_get_view(coords, &view);
    KL(color_dev::decolor_tracer, (k_cnf(n)), (view, R, Po, color, n, pp, cc, rnd.state));
}

static void ini_rnd(int n, curandState_t *state) {
    long seed = 1234567;
    KL(color_dev::ini_rnd, (k_cnf(n)), (seed, n, state));
}

void tracers_ini(const int n) {
    Dalloc(&rnd.state, n);
    ini_rnd(n, rnd.state);
}

void tracers_fin() {
    Dfree(rnd.state);
}
