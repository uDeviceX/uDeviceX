__global__ void main(float mass, Fparams fpar, int n, const Particle *pp, /**/ Force *ff) {
    enum {X, Y};
    int pid;
    float fx, fy, *f;
    float x, y, lx, ly;
    const float *r;
    const float PI = 3.141592653589793;

    pid = threadIdx.x + blockDim.x * blockIdx.x;
    if (pid >= n) return;

    r = pp[pid].r;
    f = ff[pid].f;

    lx = XS; ly = YS;
    x = r[X] + 0.5*XS; y = r[Y] + 0.5*YS;
    x /=   lx; y /=   ly;
    x *= 2*PI; y *= 2*PI;

    fx =  2*sin(x)*cos(y);
    fy = -2*cos(x)*sin(y);
    fx *= fpar.a; fy *= fpar.a;

    f[X] += mass * fx;
    f[Y] += mass * fy;
}
