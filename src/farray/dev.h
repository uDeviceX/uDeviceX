template <typename Fo, typename FA>
static __device__ void farray_add_force(const Fo *f, int i, /**/ FA a) {
    enum {X, Y, Z};
    float *af = a.ff + 3*i;
    af[X] += f->x;
    af[Y] += f->y;
    af[Z] += f->z;
}

static __device__ void farray_add_stress(const PairSFo *f, int i, /**/ FoSArray_v a) {
    enum {XX, XY, XZ, YY, YZ, ZZ};
    float *s = a.ss + 6*i;
    s[XX] += f->sxx;
    s[XY] += f->sxy;
    s[XZ] += f->sxz;
    s[YY] += f->syy;
    s[YZ] += f->syz;
    s[ZZ] += f->szz;
}

static __device__ void farray_add(const PairFo *f, int i, /**/ FoArray_v a) {
    farray_add_force(f, i, /**/ a);
}

static __device__ void farray_add(const PairSFo *f, int i, /**/ FoSArray_v a) {
    farray_add_force(f, i, /**/ a);
     farray_add_stress(f, i, /**/ a);
}


template <typename Fo, typename FA>
static __device__ void farray_atomic_add_force(const Fo *f, int i, /**/ FA a) {
    enum {X, Y, Z};
    float *af = a.ff + 3*i;
    atomicAdd(&af[X], f->x);
    atomicAdd(&af[Y], f->y);
    atomicAdd(&af[Z], f->z);
}

static __device__ void farray_atomic_add_stress(const PairSFo *f, int i, /**/ FoSArray_v a) {
    enum {XX, XY, XZ, YY, YZ, ZZ};
    float *s = a.ss + 6*i;
    atomicAdd(&s[XX], f->sxx);
    atomicAdd(&s[XY], f->sxy);
    atomicAdd(&s[XZ], f->sxz);
    atomicAdd(&s[YY], f->syy);
    atomicAdd(&s[YZ], f->syz);
    atomicAdd(&s[ZZ], f->szz);
}

static __device__ void farray_atomic_add(const PairFo *f, int i, /**/ FoArray_v a) {
    farray_atomic_add_force(f, i, /**/ a);
}

static __device__ void farray_atomic_add(const PairSFo *f, int i, /**/ FoSArray_v a) {
    farray_atomic_add_force(f, i, /**/ a);
    farray_atomic_add_stress(f, i, /**/ a);
}



static PairFo farray_fo0(FoArray_v) {
    PairFo f;
    f.x = f.y = f.z = 0;
    return f;
}

static PairSFo farray_fo0(FoSArray_v) {
    PairSFo f;
    f.x = f.y = f.z = 0;
    f.sxx = f.sxy = f.sxz = 0;
    f.syy = f.syz = f.szz = 0;
    return f;
}
