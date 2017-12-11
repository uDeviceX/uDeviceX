static __device__ float3 get_gp(Coords, FParam_cste_d par, Particle) {
    return par.a;
}

static __device__ float3 get_gp(Coords c, FParam_dp_d par, Particle p) {
    enum {X, Y, Z};
    float d, f;
    d = yl2yc(c, p.r[Y]);
    f = d > 0 ? par.a : -par.a;
    return make_float3(f, 0, 0);
}

static __device__ float3 get_gp(Coords c, FParam_shear_d par, Particle p) {
    enum {X, Y, Z};
    float d, f;
    d = yl2yc(c, p.r[Y]);
    f = d * par.a;
    return make_float3(f, 0, 0);
}

static __device__ float3 get_gp(Coords c, FParam_rol_d par, Particle p) {
    enum {X, Y, Z};
    float x, y, lx, ly;
    float3 f;
    const float PI = 3.141592653589793;
    
    lx = xdomain(c);
    ly = ydomain(c);

    x = xl2xg(c, p.r[X]);
    y = yl2yg(c, p.r[Y]);

    x *= 2*PI / lx;
    y *= 2*PI / ly;

    f.x =  2*sin(x)*cos(y) * par.a;
    f.y = -2*cos(x)*sin(y) * par.a;    
    f.z = 0;
    return f;
}

template <typename P>
__global__ void force(Coords c, P par, float mass, int n, const Particle *pp, /**/ Force *ff) {
    enum {X, Y, Z};
    int i;
    Particle p;
    float3 gp; // pressure gradient
    float *f; // force
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;

    p = pp[i];
    f = ff[i].f;
    gp = get_gp(c, par, p);

    f[X] += mass * gp.x;
    f[Y] += mass * gp.y;
    f[Z] += mass * gp.z;
}
