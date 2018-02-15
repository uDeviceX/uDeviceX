template<typename FA>
static __device__ void farray_add_common(const PairFo *f, int i, /**/ FA a) {
    enum {X, Y, Z};
    float *af = a.ff + 3*i;
    af[X] += f->x;
    af[Y] += f->y;
    af[Z] += f->z;
}

static __device__ void farray_add(const PairFo *f, int i, /**/ PaArray_v a) {
    farray_add_common(f, i, /**/ a);
}

static __device__ void farray_add(const PairFo *f, int i, /**/ PaSArray_v a) {
    enum {XX, XY, XZ, YY, YZ, ZZ};
    float *s = a.ss + 6*i;
    farray_add_common(f, i, /**/ a);
    s[XX] += f->sxx;
    s[XY] += f->sxy;
    s[XZ] += f->sxz;
    s[YY] += f->syy;
    s[YZ] += f->syz;
    s[ZZ] += f->szz;
}


template<typename FA>
static __device__ void farray_atomic_add_common(const PairFo *f, int i, /**/ FA a) {
    enum {X, Y, Z};
    float *af = a.ff + 3*i;
    atomicAdd(&af[X], f->x);
    atomicAdd(&af[Y], f->y);
    atomicAdd(&af[Z], f->z);
}

static __device__ void farray_atomic_add(const PairFo *f, int i, /**/ PaArray_v a) {
    farray_atomic_add_common(f, i, /**/ a);
}

static __device__ void farray_atomic_add(const PairFo *f, int i, /**/ PaSArray_v a) {
    enum {XX, XY, XZ, YY, YZ, ZZ};
    float *s = a.ss + 6*i;
    farray_atomic_add_common(f, i, /**/ a);
    atomicAdd(&s[XX], f->sxx);
    atomicAdd(&s[XY], f->sxy);
    atomicAdd(&s[XZ], f->sxz);
    atomicAdd(&s[YY], f->syy);
    atomicAdd(&s[YZ], f->syz);
    atomicAdd(&s[ZZ], f->szz);
}
