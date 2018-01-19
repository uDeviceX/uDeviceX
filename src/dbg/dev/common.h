static __device__ void report(err_type e) {
    if (e != ERR_NONE) atomicExch(&error, e);
}

static __device__ err_type check_float(float a) {
    if (isnan(a)) return ERR_NAN_VAL;
    if (isinf(a)) return ERR_INF_VAL;
    return ERR_NONE;
}

static __device__ err_type check_float3(const float a[3]) {
    enum {X, Y, Z};
    err_type e;
#define check_ret(A) if ((e = check_float(A)) != ERR_NONE) return e
    check_ret(a[X]);
    check_ret(a[Y]);
    check_ret(a[Z]);
#undef check_ret
    return ERR_NONE;
}
