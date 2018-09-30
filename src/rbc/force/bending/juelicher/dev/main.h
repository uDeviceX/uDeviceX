enum {X, Y, Z};
static __device__  int small(double s) {
    const double eps = 1e-12;
    if      (s >  eps) return 0;
    else if (s < -eps) return 0;
    else               return 1;
}
static __device__ void append(double x, int i, float *a) { atomicAdd(&a[i], x); }
static __device__ double tri_area(const double a[3], const double b[3], const double c[3]) { return tri_dev::kahan_area(a, b, c); }
static __device__ double tri_dih(const double a[3], const double b[3], const double c[3], const double d[3]) {
    double x, y;
    tri_dev::dihedral_xy(a, b, c, d, /**/ &x, &y);
    return -atan2(y, x); /* TODO: */
}
static __device__ void vec_scalar(const double a[3], double s, /**/ double b[3]) {
    b[X] = s*a[X]; b[Y] = s*a[Y]; b[Z] = s*a[Z];
}
static __device__ void vec_minus(const double a[3], const double b[3], /**/ double c[3]) {
    c[X] = a[X] - b[X];
    c[Y] = a[Y] - b[Y];
    c[Z] = a[Z] - b[Z];
}
static __device__ double vec_dot(const double a[3], const double b[3]) {
   return a[X]*b[X] + a[Y]*b[Y] + a[Z]*b[Z];
}
static __device__ double vec_abs(const double a[3]) { return sqrt(vec_dot(a, a)); }

static __device__ void vec_copy(const double a[3], double b[3]) {
    b[X] = a[X]; b[Y] = a[Y]; b[Z] = a[Z];
}

static __device__ void vec_negative(const double a[3], /**/ double b[3]) {
    b[X] = -a[X]; b[Y] = -a[Y]; b[Z] = -a[Z];
}
static __device__ void vec_norm(const double a[3], /**/ double b[3]) {
    double s;
    s = vec_abs(a);
    if (!small(s)) vec_scalar(a, 1/s, /**/ b);
    else vec_copy(a, b);
}

static __device__ void dedg_abs(double a[3], double b[3], /**/ double da[3], double db[3]) {
    double u[3], e[3];
    vec_minus(b, a,   u);
    vec_norm(u,   e);
    vec_copy(e,     db);
    vec_negative(e, da);
}

static __device__ void vec_scalar_append(const double a[3], double s, int i,
                                         /**/ float *f) {
    append(s*a[X], 3*i,     f);
    append(s*a[Y], 3*i + 1, f);
    append(s*a[Z], 3*i + 2, f);
}

static __device__ void get(const Particle *pp, int i, /**/ double r[3]) {
    r[X] = pp[i].r[X];
    r[Y] = pp[i].r[Y];
    r[Z] = pp[i].r[Z];
}
static __device__ void get2(const Particle *pp, int i, int j,
                            /**/ double a[3], double b[3]) {
    get(pp, i, /**/ a);
    get(pp, j, /**/ b);
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

static __device__ void force_edg0(float H0, float curva_mean_area_tot,
                                  const Particle *pp, const int4 dih,
                                  float theta, const float *lentheta, const float *area,
                                  /**/ float *f, float *fad) {
    int j, k;
    double b[3], c[3], db[3], dc[3];
    double coef;
    
    j = dih.y; k = dih.z;
    get2(pp, j, k, /**/ b, c);
    dedg_abs(b, c, db, dc);

    coef = - ( (lentheta[j]/area[j]/4 - H0) + (lentheta[k]/area[k]/4 - H0) ) * theta;
    vec_scalar_append(db, coef, j, f);
    vec_scalar_append(dc, coef, k, f);
    coef = -curva_mean_area_tot/4 * theta;
    vec_scalar_append(db, coef, j, fad);
    vec_scalar_append(dc, coef, k, fad);
}

__global__ void force_edg(int nv, int ne, int nc, float H0,
                          const Particle *pp, const int4 *dih,
                          const float *curva_mean_area_tot,
                          const float *theta, const float *lentheta, const float *area,
                          /**/ float *f, float *fad) {
    int i;
    int e, c; /* edge, cell */
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= ne*nc) return;

    c = i / ne;
    e = i % ne;

    pp       += nv*c;
    dih      += e;
    
    curva_mean_area_tot += c;
    theta    += ne*c + e;
    lentheta += nv*c;
    area     += nv*c;

    f        += 3*nv*c;
    fad      += 3*nv*c;

    force_edg0(H0, *curva_mean_area_tot,
               pp, *dih,
               *theta, lentheta, area, /**/ f, fad);
}
