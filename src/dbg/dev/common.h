enum {X, Y, Z};

static __device__ void report(err_type e) {
    if (e != err::NONE) atomicExch(&error, e);
}

static __device__ err_type check_float(float a) {
    if (isnan(a)) return err::NAN_VAL;
    if (isinf(a)) return err::INF_VAL;
    return err::NONE;
}

static __device__ err_type check_float3(const float a[3]) {
    err_type e;
#define check_ret(A) if ((e = check_float(A)) != err::NONE) return e
    check_ret(a[0]);
    check_ret(a[1]);
    check_ret(a[2]);
#undef check_ret
    return err::NONE;
}
