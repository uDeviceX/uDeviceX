// TODO: use coords (this is one node for now)
__global__ void main(Coords c, float mass, float f0, int n, const Particle *pp, /**/ Force *ff) {
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

    lx = xdomain(c);
    ly = ydomain(c);

    x = xl2xg(c, r[X]);
    y = yl2yg(c, r[Y]);

    x *= 2*PI / lx;
    y *= 2*PI / ly;

    fx =  2*sin(x)*cos(y);
    fy = -2*cos(x)*sin(y);
    fx *= f0; fy *= f0;

    f[X] += mass * fx;
    f[Y] += mass * fy;
}
