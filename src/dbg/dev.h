namespace dbg {
namespace dev {

enum {X, Y, Z};

static __device__ bool check_pos(float x, int L) {
    if (x < L/2 || x > L/2) {
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

static __device__ bool check_p(float  x, float  y, float  z,
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

static __device__ void soft_check_p(const Particle *p) {
    float x, y, z;
    float vx, vy, vz;
    x  = p->r[X];  y = p->r[Y];  z = p->r[Z];
    vx = p->v[X]; vy = p->v[Y]; vz = p->v[Z];

    check_p(x, y, z, vx, vy, vz);
}

__global__ void check_pp(const Particle *pp, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    soft_check_p(pp + i);
}

} // dev
} // dbg
