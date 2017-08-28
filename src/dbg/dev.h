namespace dbg {
namespace dev {

enum {X, Y, Z};

static __device__ err_type valid_real(float x) {
    if (isnan(x)) return err::NAN_VAL;
    if (isinf(x)) return err::INF_VAL;
    return err::NONE;
}

static __device__ bool valid_pos(float x, int L, bool verbose) {
    if (x < -L/2 || x > L/2) {
        if (verbose) printf("DBG: x = %g (L = %d)\n", x, L);
        return false;
    }
    return true;
}

static __device__ bool valid_vel(float v, int L, bool verbose) {
    float dx = fabs(v * dt);
    if (dx > L/2) {
        if (verbose) printf("DBG: v = %g (L = %d)\n", v, L);
        return false;
    }
    return true;
}

static __device__ bool valid_unpacked_p(float  x, float  y, float  z,
                                        float vx, float vy, float vz, bool verbose) {
    bool ok = true;
    ok &= valid_pos(x, XS, verbose);
    ok &= valid_pos(y, YS, verbose);
    ok &= valid_pos(z, ZS, verbose);

    ok &= valid_vel(vx, XS, verbose);
    ok &= valid_vel(vy, YS, verbose);
    ok &= valid_vel(vz, ZS, verbose);
    
    return ok;
}

static __device__ bool valid_p(const Particle *p, bool verbose) {
    float x, y, z;
    float vx, vy, vz;
    x  = p->r[X];  y = p->r[Y];  z = p->r[Z];
    vx = p->v[X]; vy = p->v[Y]; vz = p->v[Z];

    return valid_unpacked_p(x, y, z, vx, vy, vz, verbose);
}

static __global__ void check_pp(const Particle *pp, int n, bool verbose = false) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    if (!valid_p(pp + i, verbose)) atomicExch(&error, err::INVALID);
}

static __device__ bool valid_unpacked_p_pu(float x, float y, float z, bool verbose) {
    bool ok = true;
    ok &= valid_pos(x, 3*XS, verbose);
    ok &= valid_pos(y, 3*YS, verbose);
    ok &= valid_pos(z, 3*ZS, verbose);

    return ok;
}

static __device__ bool valid_vel3(float vx, float vy, float vz, bool verbose) {
    bool ok = true;
    ok &= valid_vel(vx, XS, verbose);
    ok &= valid_vel(vy, YS, verbose);
    ok &= valid_vel(vz, ZS, verbose);

    return ok;
}

static __device__ bool valid_p_pu(const Particle *p, bool verbose) {
    float x, y, z;
    x  = p->r[X];  y = p->r[Y];  z = p->r[Z];

    return valid_unpacked_p_pu(x, y, z, verbose);
}

static __device__ bool valid_vv(const Particle *p, bool verbose) {
    const float *v = p->v;
    return valid_vel3(v[X], v[Y], v[Z], verbose);
}

static __global__ void check_pp_pu(const Particle *pp, int n, bool verbose = false) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    if (!valid_p_pu(pp + i, verbose)) atomicExch(&error, err::INVALID);
}

static __global__ void check_vv(const Particle *pp, int n, bool verbose = false) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    if (!valid_vv(pp + i, verbose)) atomicExch(&error, err::INVALID);
}

static __device__ bool valid_acc(float a, int L, bool verbose) {
    float dx = fabs(a * dt * dt);
    if (dx > L/2) {
        if (verbose) printf("DBG: a = %g (L = %d)\n", a, L);
        return false;
    }
    return true;
}

static __device__ bool valid_unpacked_f(float fx, float fy, float fz, bool verbose) {
    bool ok = true;
    ok &= valid_acc(fx, XS, verbose);
    ok &= valid_acc(fy, YS, verbose);
    ok &= valid_acc(fz, ZS, verbose);

    return ok;
}

static __device__ bool valid_f(const Force *f, bool verbose) {
    float fx, fy, fz;
    fx = f->f[X]; fy = f->f[Y]; fz = f->f[Z];

    return valid_unpacked_f(fx, fy, fz, verbose);
}

static __global__ void check_ff(const Force *ff, int n, bool verbose = false) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    if(!valid_f(ff + i, verbose)) atomicExch(&error, err::INVALID);
}

} // dev
} // dbg
