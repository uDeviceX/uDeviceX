namespace dbg {
namespace dev {

enum {X, Y, Z};

static __device__ void check_pos(float x, int L) {
    if (x < L/2 || x > L/2) printf("DBG: x = %g (L = %d)\n", x, L);
}

static __device__ void check_vel(float v, int L) {
    float dx = fabs(v * dt);
    if (dx > L/2) printf("DBG: v = %g (L = %d)\n", v, L);
}

static __device__ void check_p(const Particle *p) {
    float x, y, z;
    float vx, vy, vz;
    x  = p->r[X];  y = p->r[Y];  z = p->r[Z];
    vx = p->v[X]; vy = p->v[Y]; vz = p->v[Z];

    check_pos(x, XS);
    check_pos(y, YS);
    check_pos(z, ZS);

    check_vel(vx, XS);
    check_vel(vy, YS);
    check_vel(vz, ZS);    
}

__global__ void check_pp(const Particle *pp, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    check_p(pp + i);
}

} // dev
} // dbg
