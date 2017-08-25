namespace dbg {
namespace dev {

enum {X, Y, Z};

static __device__ bool valid_pos(float x, int L) {
    if (x < -L/2 || x > L/2) {
        printf("DBG: x = %g (L = %d)\n", x, L);
        return false;
    }
    return true;
}

static __device__ bool valid_vel(float v, int L) {
    float dx = fabs(v * dt);
    if (dx > L/2) {
        printf("DBG: v = %g (L = %d)\n", v, L);
        return false;
    }
    return true;
}

static __device__ bool valid_unpacked_p(float  x, float  y, float  z,
                                        float vx, float vy, float vz) {
    bool ok = true;
    ok &= valid_pos(x, XS);
    ok &= valid_pos(y, YS);
    ok &= valid_pos(z, ZS);

    ok &= valid_vel(vx, XS);
    ok &= valid_vel(vy, YS);
    ok &= valid_vel(vz, ZS);    

    return ok;
}

static __device__ bool valid_p(const Particle *p) {
    float x, y, z;
    float vx, vy, vz;
    x  = p->r[X];  y = p->r[Y];  z = p->r[Z];
    vx = p->v[X]; vy = p->v[Y]; vz = p->v[Z];

    return valid_unpacked_p(x, y, z, vx, vy, vz);
}

__global__ void check_pp(const Particle *pp, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    if (!valid_p(pp + i)) atomicExch(&error, err::INVALID);
}

static __device__ bool valid_unpacked_p_pu(float  x, float  y, float  z) {
    bool ok = true;
    ok &= valid_pos(x, 3*XS);
    ok &= valid_pos(y, 3*YS);
    ok &= valid_pos(z, 3*ZS);

    return ok;
}

static __device__ bool valid_p_pu(const Particle *p) {
    float x, y, z;
    x  = p->r[X];  y = p->r[Y];  z = p->r[Z];

    return valid_unpacked_p_pu(x, y, z);
}

__global__ void check_pp_pu(const Particle *pp, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    if (!valid_p_pu(pp + i)) atomicExch(&error, err::INVALID);
}


static __device__ bool valid_acc(float a, int L) {
    float dx = fabs(a * dt * dt);
    if (dx > L/2) {
        printf("DBG: a = %g (L = %d)\n", a, L);
        return false;
    }
    return true;
}

static __device__ bool valid_unpacked_f(float fx, float fy, float fz) {
    bool ok = true;
    ok &= valid_acc(fx, XS);
    ok &= valid_acc(fy, YS);
    ok &= valid_acc(fz, ZS);

    return ok;
}

static __device__ bool valid_f(const Force *f) {
    float fx, fy, fz;
    fx = f->f[X]; fy = f->f[Y]; fz = f->f[Z];

    return valid_unpacked_f(fx, fy, fz);
}

__global__ void check_ff(const Force *ff, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    if(!valid_f(ff + i)) atomicExch(&error, err::INVALID);
}

} // dev
} // dbg
