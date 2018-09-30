static __device__ double tri_area(const double a[3], const double b[3], const double c[3]) { return tri_dev::kahan_area(a, b, c); }
static __device__ double tri_dih(const double a[3], const double b[3], const double c[3], const double d[3]) {
    double x, y;
    tri_dev::dihedral_xy(a, b, c, d, /**/ &x, &y);
    return -atan2(y, x); /* TODO: */
}
static __device__ void vec_minus(const double a[3], const double b[3], /**/ double c[3]) {
    enum {X, Y, Z};
    c[X] = a[X] - b[X];
    c[Y] = a[Y] - b[Y];
    c[Z] = a[Z] - b[Z];
}
static __device__ double vec_dot(const double a[3], const double b[3]) {
    enum {X, Y, Z};
   return a[X]*b[X] + a[Y]*b[Y] + a[Z]*b[Z];
}
static __device__ double vec_abs(const double a[3]) { return sqrt(vec_dot(a, a)); }
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
static __device__ void get4(const Particle *pp, int i, int j, int k, int l,
                            /**/ double a[3], double b[3], double c[3], double d[3]) {
    get(pp, i, /**/ a);
    get(pp, j, /**/ b);
    get(pp, k, /**/ c);
    get(pp, l, /**/ d);
}

__global__ void sum(int nv, const float *from, /**/ float *to) {
    int i, c;
    float s;
    i  = threadIdx.x + blockIdx.x * blockDim.x;
    c  = blockIdx.y;
    if (i < nv) s = from[c*nv + i]; else s = 0;
    s = warpReduceSum(s);
    if ((threadIdx.x % warpSize) == 0) atomicAdd(&to[c], s);
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

static __device__ void compute_theta_len0(const Particle *pp, int4 dih,
                                          /**/ float *theta, float *lentheta) {
    int i, j, k, l;
    double a[3], b[3], c[3], d[3], u[3];
    double len0, theta0, lentheta0;

    i = dih.x; j = dih.y; k = dih.z; l = dih.w;
    get4(pp, i, j, k, l, /**/ a, b, c, d);

    *theta = theta0 = tri_dih(a, b, c, d);
    vec_minus(c, b, u);
    len0 = vec_abs(u);
    lentheta0 = len0*theta0;

    append(lentheta0, j,  lentheta);
    append(lentheta0, k,  lentheta);
}

__global__ void compute_theta_len(int nv, int ne, int nc,
                                  const Particle *pp, const int4 *dih,
                                  /**/ float *theta, float *lentheta) {
    int i;
    int e, c; /* edge, cell */
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= ne*nc) return;

    c = i / ne;
    e = i % ne;

    pp       += nv*c;
    theta    += ne*c + e;
    lentheta += nv*c;
    dih      += e;

    compute_theta_len0(pp, *dih, /**/ theta, lentheta);
}

__global__ void compute_mean_curv(int nc, float H0, float kb,
                                  const float *lentheta, const float *area,
                                  /**/ float *mean) {
    int i;
    float kad, pi;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= nc) return;
    pi = 3.141592653589793;
    kad = 2*kb/pi;
    mean[i] = (lentheta[i]/4 - H0 * area[i])*(4*kad*pi/area[i]);
}
