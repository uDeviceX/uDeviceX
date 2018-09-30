struct Part { double3 r, v; };
typedef double3 Pos;

static __device__ double tri_area(const double a[3], const double b[3], const double c[3]) { return tri_dev::kahan_area(a, b, c); }
static __device__ void append(double x, int i, float *a) { atomicAdd(&a[i], x); }
static __device__ void get(const Particle *pp, int i, /**/ double r[3]) {
    enum {X, Y, Z};
    r[X] = pp[i].r[X];
    r[Y] = pp[i].r[Y];
    r[Z] = pp[i].r[Z];
}

static __device__ void get3(const Particle *pp, int i, int j, int k,
                            /**/ double a[3], double b[3], double c[3]) {
    get(pp, i, /**/ a);
    get(pp, j, /**/ b);
    get(pp, k, /**/ c);
}

static __device__ void compute_area0(const Particle *pp, int4 tri, /**/ float *area) {
    double area0;
    double a[3], b[3], c[3];
    int i, j, k;
    i = tri.x; j = tri.y; k = tri.z;
    get3(pp, i, j, k, /**/ a, b, c);
    area0 = tri_area(a, b, c)/3;

    append(area0, i, area);
    append(area0, j, area);
    append(area0, k, area);
}

__global__ void compute_area(int nv, int nt, int nc,
                             const Particle *pp, const int4 *tri,
                             /**/ float *area) {
    int i;
    int t, c; /* triangle, cell */
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= nt*nc) return;

    c = i / nt;
    t = i % nt;

    pp   += nv*c;
    area += nv*c;
    tri  += t;

    compute_area0(pp, *tri, /**/ area);
}
