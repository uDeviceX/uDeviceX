namespace dbg {
namespace dev {

enum {X, Y, Z};

static __device__ bool check_pos(float x, int L) {
    if (x < -L/2 || x > L/2) {
        printf("DBG: x = %g (L = %d)\n", x, L);
        return false;
    }
    return true;
}

static __device__ bool check_vel(float v, int L) {
    float dx = fabs(v * dt);
    if (dx > L/2) {
        printf("DBG: v = %g (L = %d)\n", v, L);
        return false;
    }
    return true;
}

static __device__ bool check_unpacked_p(float  x, float  y, float  z,
                                        float vx, float vy, float vz) {
    bool ok = true;
    ok &= check_pos(x, XS);
    ok &= check_pos(y, YS);
    ok &= check_pos(z, ZS);

    ok &= check_vel(vx, XS);
    ok &= check_vel(vy, YS);
    ok &= check_vel(vz, ZS);    

    return ok;
}

static __device__ bool check_p(const Particle *p) {
    float x, y, z;
    float vx, vy, vz;
    x  = p->r[X];  y = p->r[Y];  z = p->r[Z];
    vx = p->v[X]; vy = p->v[Y]; vz = p->v[Z];

    return check_unpacked_p(x, y, z, vx, vy, vz);
}

__global__ void hard_check_pp(const Particle *pp, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    if (!check_p(pp + i)) atomicExch(&error, err::INVALID);
}


static __device__ bool check_acc(float a, int L) {
    float dx = fabs(a * dt * dt);
    if (dx > L/2) {
        printf("DBG: a = %g (L = %d)\n", a, L);
        return false;
    }
    return true;
}

static __device__ bool check_unpacked_f(float fx, float fy, float fz) {
    bool ok = true;
    ok &= check_acc(fx, XS);
    ok &= check_acc(fy, YS);
    ok &= check_acc(fz, ZS);

    return ok;
}

static __device__ bool check_f(const Force *f) {
    float fx, fy, fz;
    fx = f->f[X]; fy = f->f[Y]; fz = f->f[Z];

    return check_unpacked_f(fx, fy, fz);
}

__global__ void hard_check_ff(const Force *ff, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    assert(check_f(ff + i));
}

} // dev
} // dbg
